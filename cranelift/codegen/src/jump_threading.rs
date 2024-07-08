//! Jump Thread analysis and optimization pass.

use alloc::vec::Vec;
use smallvec::{smallvec, SmallVec};

use crate::cursor::{Cursor, FuncCursor};
use crate::dominator_tree::DominatorTree;
use crate::flowgraph::{BlockPredecessor, ControlFlowGraph};
use crate::inst_predicates::has_side_effect;
use crate::ir::InstInserterBase;
use crate::ir::{Block, BlockCall, Function, InstBuilder, InstructionData, Opcode, Value};
use crate::loop_analysis::LoopAnalysis;
use crate::trace;
use core::fmt;
use std::collections::HashMap;

/// Represents a single action to be performed as part of the jump thread analysis.
#[derive(Debug, PartialEq, Clone)]
enum JumpThreadAction {
    /// Evaluates the possible actions that we are able to take on this block
    /// and pushes them into the action queue.
    Analyze(Block),

    /// Deletes this block from the function
    Delete(Block),

    /// Merges a successor block into a predecessor block.
    ///
    /// This deletes the terminator instruction from the predecessor
    /// block, and replaces it with the successor's one. If there
    /// is any information to be gained by that transformation it will
    /// be added to the block. (i.e. the terminator was a br_if and we
    /// now know the value of the branch condition).
    MergeIntoPredecessor {
        successor: Block,
        predecessor: BlockPredecessor,
    },

    /// Replace the terminator for this block with a jump into the
    /// provided block call.
    ReplaceWithJump(Block, BlockCall),

    /// When the terminator for a block is a br_if where both sides are
    /// equal in destination, we replace the differing block args with
    /// `select`'s and inline the block.
    SelectifyBrIf(Block),

    /// This action replaces all instances of a block call with another block call
    /// to a different block.
    ///
    /// `caller_block` is the block that we are going to be modifying, it should contain
    /// a terminator with at least one instance of `inlined_blockcall`.
    ///
    /// `inlined_blockcall` should be a blockcall to a block that contains a `jump` terminator.
    /// We then inline the body of the `inlined_blockcall` block into `caller_block` and
    /// subsequently replace all instances of `inlined_blockcall` with the terminator
    /// for the `jump` instruction.
    InlineBlockCall {
        caller_block: Block,
        inlined_blockcall: BlockCall,
    },
}

impl fmt::Display for JumpThreadAction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            JumpThreadAction::Analyze(block) => write!(f, "analyze {block}"),
            JumpThreadAction::Delete(block) => write!(f, "delete {block}"),
            JumpThreadAction::MergeIntoPredecessor {
                successor,
                predecessor,
            } => write!(f, "merge {successor} into {}", predecessor.block),
            JumpThreadAction::ReplaceWithJump(block, _call) => {
                write!(f, "replace {block} terminator with jump")
            }
            JumpThreadAction::SelectifyBrIf(block) => {
                write!(f, "selectify {block}")
            }
            JumpThreadAction::InlineBlockCall { caller_block, .. } => {
                // TODO: We should print out this call
                write!(f, "inline block_call? on {caller_block}")
            }
        }
    }
}

impl JumpThreadAction {
    fn run<'a>(self, jt: &mut JumpThreadingPass<'a>) {
        match self {
            JumpThreadAction::Analyze(block) => {
                // It may happen that we try to analyze a block but it has been previously deleted
                // due to effects on other blocks.
                if !jt.func.layout.is_block_inserted(block) {
                    return;
                }

                let mut actions = jt.analyze_block(block);

                // The actions above directly modify, but do not include any reanalyze actions, so we add
                // those now.
                let analyze_actions: SmallVec<[_; 8]> = actions
                    .iter()
                    .flat_map(|action| action.analyze_actions(jt))
                    .collect();
                actions.extend(analyze_actions);

                // Debug print the actions that we performed
                #[cfg(feature = "trace-log")]
                match actions.as_slice() {
                    [] => trace!("Evaluating {block}: skip"),
                    multi => {
                        trace!("Evaluating {block}:");
                        for action in multi {
                            trace!("\t- {action}");
                        }
                    }
                };

                // The actions queue is always popped from the last place, so we need
                // to reverse this set of action so that they get executed in their intended order.
                actions.reverse();
                jt.actions.extend(actions.into_iter());
            }
            JumpThreadAction::Delete(block) => {
                // Remove all instructions from `block`.
                while let Some(inst) = jt.func.layout.first_inst(block) {
                    jt.func.layout.remove_inst(inst);
                }

                // Once the block is completely empty, we can update the CFG which removes it from any
                // predecessor lists.
                jt.cfg.recompute_block(jt.func, block);

                // Finally remove the block
                jt.func.layout.remove_block(block);
            }
            JumpThreadAction::MergeIntoPredecessor {
                successor,
                predecessor,
            } => {
                let succ = successor;
                let pred = predecessor.block;
                let pred_inst = predecessor.inst;

                debug_assert!(jt.cfg.pred_iter(succ).all(|b| b.block == pred));
                debug_assert!(jt.func.layout.is_block_inserted(succ));
                debug_assert!(jt.func.layout.is_block_inserted(pred));
                debug_assert_eq!(jt.cfg.pred_iter(succ).count(), 1);
                debug_assert_eq!(jt.cfg.succ_iter(pred).count(), 1);

                // If the branch instruction that lead us to this block wasn't an unconditional jump, then
                // we have a conditional jump sequence that we should not break.
                let branch_dests =
                    jt.func.dfg.insts[pred_inst].branch_destination(&jt.func.dfg.jump_tables);

                debug_assert_eq!(branch_dests.len(), 1);

                let branch_args = branch_dests[0]
                    .args_slice(&jt.func.dfg.value_lists)
                    .to_vec();

                // TODO: should we free the entity list associated with the block params?
                let block_params = jt
                    .func
                    .dfg
                    .detach_block_params(succ)
                    .as_slice(&jt.func.dfg.value_lists)
                    .to_vec();

                debug_assert_eq!(block_params.len(), branch_args.len());

                // If there were any block parameters in block, then the last instruction in pred will
                // fill these parameters. Make the block params aliases of the terminator arguments.
                for (block_param, arg) in block_params.into_iter().zip(branch_args) {
                    if block_param != arg {
                        jt.func.dfg.change_to_alias(block_param, arg);
                    }
                }

                let layout = &mut jt.func.layout;
                // Remove the terminator branch to the current block.
                layout.remove_inst(pred_inst);

                // Move all the instructions to the predecessor.
                while let Some(inst) = layout.first_inst(succ) {
                    layout.remove_inst(inst);
                    layout.append_inst(inst, pred);
                }

                // If succ was cold, pred is now also cold. Except if it's the entry
                // block which cannot be cold.
                let pred_is_entry = layout.entry_block().unwrap() == pred;
                if layout.is_cold(succ) && !pred_is_entry {
                    layout.set_cold(pred);
                }

                // Now that we are done, we should update the successors of the pred block
                jt.cfg.recompute_block(jt.func, pred);
            }
            JumpThreadAction::ReplaceWithJump(block, new_target) => {
                let target_block = new_target.block(&jt.func.dfg.value_lists).clone();
                let target_values = new_target.args_slice(&jt.func.dfg.value_lists).to_vec();

                // Remove the terminator instruction on the block.
                let terminator = jt.func.layout.last_inst(block).unwrap();
                let terminator_srcloc = jt.func.srcloc(terminator);
                jt.func.layout.remove_inst(terminator);

                // Insert the new terminator as the last instruction
                let mut cursor = FuncCursor::new(jt.func).at_bottom(block);
                cursor.set_srcloc(terminator_srcloc);
                cursor.ins().jump(target_block, &target_values[..]);

                // We may have changed the successors of this function, and we definitley changed
                // the terminator instruction so we need to recompute the cfg for this block
                jt.cfg.recompute_block(jt.func, block);
            }
            JumpThreadAction::SelectifyBrIf(block) => {
                let terminator = jt.func.layout.last_inst(block).unwrap();
                let (brif_cond, branch_dests) = match jt.func.dfg.insts[terminator] {
                    InstructionData::Brif { arg, blocks, .. } => (arg, blocks),
                    _ => unreachable!("expected br_if"),
                };
                let target_block = branch_dests[0].block(&jt.func.dfg.value_lists);
                let true_args = branch_dests[0]
                    .args_slice(&jt.func.dfg.value_lists)
                    .to_vec();
                let false_args = branch_dests[1]
                    .args_slice(&jt.func.dfg.value_lists)
                    .to_vec();

                // Delete the terminator instruction
                jt.func.layout.remove_inst(terminator);

                // Build a select for each different value
                let mut cursor = FuncCursor::new(jt.func).at_bottom(block);
                let call_args: Vec<_> = true_args
                    .iter()
                    .zip(false_args.iter())
                    .map(|(&true_arg, &false_arg)| {
                        if true_arg == false_arg {
                            true_arg
                        } else {
                            cursor.ins().select(brif_cond, true_arg, false_arg)
                        }
                    })
                    .collect();

                // Transform the terminator into a jump
                cursor.ins().jump(target_block, &call_args[..]);

                // Finally recompute this block since we just changd the terminator
                jt.cfg.recompute_block(jt.func, block);
            }
            JumpThreadAction::InlineBlockCall {
                caller_block,
                inlined_blockcall,
            } => {
                let inlined_block = inlined_blockcall.block(&jt.func.dfg.value_lists).clone();
                let inlined_call_values = inlined_blockcall
                    .args_slice(&jt.func.dfg.value_lists)
                    .to_vec();
                let inlined_block_terminator = jt.func.layout.last_inst(inlined_block).unwrap();
                let inlined_block_branch_destinations = jt.func.dfg.insts[inlined_block_terminator]
                    .branch_destination(&jt.func.dfg.jump_tables);
                debug_assert_eq!(inlined_block_branch_destinations.len(), 1);

                let new_target_block =
                    inlined_block_branch_destinations[0].block(&jt.func.dfg.value_lists);
                let new_target_values = inlined_block_branch_destinations[0]
                    .args_slice(&jt.func.dfg.value_lists)
                    .to_vec();

                // In order to insert instructions into the block we first remove the
                // current terminator from the block
                let terminator = jt.func.layout.last_inst(caller_block).unwrap();
                jt.func.layout.remove_inst(terminator);

                // Inline all instructions from the call block, into the current block.
                let block_to_call_map = {
                    let mut cursor = FuncCursor::new(jt.func).at_bottom(caller_block);
                    let map = Self::copy_block_instructions(
                        &mut cursor,
                        inlined_block,
                        &inlined_call_values[..],
                    );
                    // The function above also copied the terminator for the call block, which we do not
                    // want, so let remove that one.
                    cursor.prev_inst();
                    cursor.remove_inst();
                    map
                };

                // Prepare a set of values that we will replace the old block call with.
                let new_target_values: Vec<_> = new_target_values
                    .into_iter()
                    .map(|val| block_to_call_map[&val])
                    .collect();

                // Update the terminator and replace all of the blockcalls into the new target
                // block call.
                let dfg = &mut jt.func.dfg;
                let terminator_instdata = &mut dfg.insts[terminator];
                let jump_tables = &mut dfg.jump_tables;
                let value_lists = &mut dfg.value_lists;
                
                for dest in terminator_instdata
                    .branch_destination_mut(jump_tables)
                    .iter_mut()
                {
                    let block = dest.block(value_lists);
                    let args = dest.args_slice(value_lists);

                    if block == inlined_block && args == inlined_call_values {
                        dest.set_block(new_target_block, value_lists);
                        dest.clear(value_lists);
                        dest.extend(new_target_values.iter().copied(), value_lists);
                    }
                }
                jt.func.layout.append_inst(terminator, caller_block);

                // We have changed the sucessors of this function, so we now get to recompute it.
                jt.cfg.recompute_block(jt.func, caller_block);
            }
        }
    }

    fn analyze_actions(&self, jt: &JumpThreadingPass<'_>) -> SmallVec<[JumpThreadAction; 8]> {
        let mut actions = SmallVec::new();

        match self {
            JumpThreadAction::Analyze(_) => {}
            JumpThreadAction::Delete(block) => {
                // The successors of this block may now be dead, so we should reanalyze them.
                actions.extend(
                    jt.cfg
                        .succ_iter(*block)
                        .filter(|succ| succ != block)
                        // Filter for blocks with one or fewer predecessors. It's only one, because
                        // we haven't yet deleted the predecessor and updated the CFG. Once
                        // this analyze runs it should become 0 predecessors.
                        .filter(|succ| jt.cfg.pred_iter(*succ).count() <= 1)
                        .map(JumpThreadAction::Analyze),
                );
            }
            JumpThreadAction::MergeIntoPredecessor { predecessor, .. } => {
                // The only remaining block after this operation is the preecessor.
                actions.push(JumpThreadAction::Analyze(predecessor.block));
            }
            JumpThreadAction::ReplaceWithJump(block, _) => {
                // We need to reanalyze both this block, and the previous successors. This block
                // may now have other threading opportunities, and the previous successors
                // may now have become dead code that we should eliminate.
                actions.extend(
                    jt.cfg
                        .succ_iter(*block)
                        .filter(|block| {
                            // We only want to reanalyze blocks that should be deleted, which means that
                            // they must have only one predecessor before this transform. (which is us)
                            //
                            // When considering the predecessors we want to take special attention to blocks
                            // that reference themselves, since they have one extra predecessor that we
                            // don't want to count.
                            let pred_count = jt
                                .cfg
                                .pred_iter(*block)
                                .filter(|pred| pred.block != *block)
                                .count();

                            pred_count == 1
                        })
                        .chain(Some(*block))
                        .map(JumpThreadAction::Analyze),
                );
            }
            JumpThreadAction::SelectifyBrIf(block) => {
                // Reanalyze ourselves since there may be further optimization opportunities.
                actions.push(JumpThreadAction::Analyze(*block));
                // Reanalyze the successors since they may have become dead.
                actions.extend(jt.cfg.succ_iter(*block).map(JumpThreadAction::Analyze));
            }
            JumpThreadAction::InlineBlockCall { caller_block, .. } => {
                // Reanalyze the `caller`, since we just changed it's terminator
                actions.push(JumpThreadAction::Analyze(*caller_block));
            }
        };

        // Make sure we don't accidentally analyze the same block multiple times
        actions.dedup();

        return actions;
    }

    /// Clone all block instructions (except the terminator) into the target FuncCursor
    /// call_args is the list of values that this block would have been called with
    /// and provides the inital point to translate values.
    ///
    /// This function returns a map of the values in the original block, into the
    /// new values that were inlined.
    fn copy_block_instructions(
        cursor: &mut FuncCursor,
        block: Block,
        call_args: &[Value],
    ) -> HashMap<Value, Value> {
        // Start our map with the block params, these map to the original values
        // in the block call.
        let block_params = cursor.func.dfg.block_params(block);
        let mut block_to_call_map: HashMap<_, _> = block_params
            .into_iter()
            .zip(call_args)
            .map(|(b, c)| (*b, *c))
            .collect();

        // We have to clone this into a vec so that we don't borrow the func layout
        // while iterating it.
        let block_insts: Vec<_> = cursor.func.layout.block_insts(block).collect();
        for block_inst in block_insts.into_iter() {
            let new_inst = cursor.func.dfg.clone_inst(block_inst);

            // Translate all values in inst with new val
            cursor.func.dfg.resolve_inst_aliases(new_inst);
            cursor
                .func
                .dfg
                .map_inst_values(new_inst, |src_val| block_to_call_map[&src_val]);

            cursor.set_srcloc(cursor.func.srcloc(block_inst));
            cursor.insert_built_inst(new_inst);

            let old_results = cursor.func.dfg.inst_results(block_inst);
            let new_results = cursor.func.dfg.inst_results(new_inst);
            block_to_call_map.extend(old_results.into_iter().zip(new_results));
        }

        block_to_call_map
    }
}

pub struct JumpThreadingConfig {
    /// When inlining blocks, how many "new" duplicate instructions are acceptable?
    max_inline_cost: u32,
}

impl Default for JumpThreadingConfig {
    fn default() -> Self {
        Self {
            max_inline_cost: 10,
        }
    }
}

pub struct JumpThreadingPass<'a> {
    /// The function we're operating on.
    func: &'a mut Function,
    /// Control flow graph
    cfg: &'a mut ControlFlowGraph,
    /// Dominator tree for the CFG, used to visit blocks in pre-order
    /// so we see value definitions before their uses, and also used for
    /// O(1) dominance checks.
    domtree: &'a mut DominatorTree, //Preorder,
    // domtree_preorder: DominatorTreePreorder,
    /// Loop analysis results. We generally avoid performing actions
    /// on blocks belonging to loops since that can generate irreducible
    /// control flow
    loop_analysis: &'a mut LoopAnalysis,

    /// A queue of actions that we have pending
    actions: Vec<JumpThreadAction>,

    /// A series of knobs that we can use to tune how this pass works.
    config: JumpThreadingConfig,
}

impl<'a> JumpThreadingPass<'a> {
    pub fn new(
        func: &'a mut Function,
        cfg: &'a mut ControlFlowGraph,
        domtree: &'a mut DominatorTree,
        loop_analysis: &'a mut LoopAnalysis,
    ) -> Self {
        Self {
            func,
            cfg,
            domtree,
            loop_analysis,
            actions: Vec::new(),
            config: JumpThreadingConfig::default(),
        }
    }

    pub fn run(&mut self) {
        // Start by analyzing all blocks
        self.actions
            .extend(self.func.layout.blocks().map(JumpThreadAction::Analyze));

        // Run actions until we are done
        while let Some(action) = self.actions.pop() {
            action.run(self);
        }

        // Now that we're done, rebuild whatever structures might be necessary
        // The CFG is always kept up to date, so we don't need to rebuild it here.
        self.domtree.clear();
        self.domtree.compute(self.func, self.cfg);
        self.loop_analysis.clear();
        self.loop_analysis
            .compute(self.func, self.cfg, self.domtree);
    }

    fn analyze_block(&mut self, block: Block) -> SmallVec<[JumpThreadAction; 8]> {
        let terminator = self.func.layout.last_inst(block).unwrap();
        let terminator_opcode = self.func.dfg.insts[terminator].opcode();

        let is_entry_block = self.func.layout.entry_block().unwrap() == block;
        let succ_count = self.cfg.succ_iter(block).count();
        let pred_count = self.cfg.pred_iter(block).count();
        let is_self_succ = self.cfg.succ_iter(block).any(|succ| succ == block);

        // During other transformations we may have ended up in a situation where this block
        // is now unreachable. So we should delete it.
        let is_unreachable = pred_count == 0 && !is_entry_block;
        let is_infinite_loop = is_self_succ && pred_count == 1 && succ_count == 1;
        if is_unreachable || is_infinite_loop {
            return smallvec![JumpThreadAction::Delete(block)];
        }

        // If all of our terminator block calls are the same we can replace it with a jump terminator.
        // This can unlock further opportunities for optimization so we should reevaluate this block again.
        let branch_dests =
            self.func.dfg.insts[terminator].branch_destination(&self.func.dfg.jump_tables);
        let all_dest_blocks_equal = branch_dests.windows(2).all(|bc| {
            let lhs_block = bc[0].block(&self.func.dfg.value_lists);
            let rhs_block = bc[1].block(&self.func.dfg.value_lists);
            lhs_block == rhs_block
        });
        let all_dest_args_equal = branch_dests.windows(2).all(|bc| {
            let lhs_args = bc[0].args_slice(&self.func.dfg.value_lists);
            let rhs_args = bc[1].args_slice(&self.func.dfg.value_lists);
            lhs_args == rhs_args
        });
        let all_dests_equal = all_dest_blocks_equal && all_dest_args_equal;
        if branch_dests.len() > 1 && all_dests_equal {
            return smallvec![JumpThreadAction::ReplaceWithJump(block, branch_dests[0])];
        }

        // Const eval brif and br_table's
        // Since our main const eval pass is part of egraphs, this is unlikely to have
        // much of an effect. It may still be useful for frontends other than
        // wasmtime or if we one day run these passes multiple times.
        if matches!(terminator_opcode, Opcode::Brif | Opcode::BrTable) {
            let branch_value = match self.func.dfg.insts[terminator] {
                InstructionData::Brif { arg, .. } => arg,
                InstructionData::BranchTable { arg, .. } => arg,
                _ => unreachable!(),
            };

            if let Some(val) = self.i64_from_iconst(branch_value) {
                let target_blockcall = match self.func.dfg.insts[terminator] {
                    InstructionData::Brif { blocks, .. } if val != 0 => blocks[0],
                    InstructionData::Brif { blocks, .. } if val == 0 => blocks[1],
                    InstructionData::BranchTable { table, .. } => {
                        let table = &self.func.stencil.dfg.jump_tables[table];
                        let non_default_blocks = table.as_slice();

                        non_default_blocks
                            .get(val as usize)
                            .copied()
                            .unwrap_or_else(|| table.default_block())
                    }
                    _ => unreachable!(),
                };

                return smallvec![JumpThreadAction::ReplaceWithJump(block, target_blockcall)];
            }
        }

        // If we only have one successor, and that block only has one predecessor
        // we most definitley want to merge into it. We also check if the terminator
        // is a jump instruction, even with a single  successor, we can still have
        // differing block args, which has to be handled differently.
        //
        // We also don't need to worry about computing the cost of this merger
        // since it can only reduce the total cost of the function.
        if succ_count == 1 && terminator_opcode == Opcode::Jump {
            let succ = self.cfg.succ_iter(block).nth(0).unwrap();
            let succ_predecessors = self.cfg.pred_iter(succ).count();
            if succ_predecessors == 1 && block != succ {
                let merge_pred = self.cfg.pred_iter(succ).nth(0).unwrap();

                return smallvec![
                    JumpThreadAction::MergeIntoPredecessor {
                        successor: succ,
                        predecessor: merge_pred,
                    },
                    JumpThreadAction::Delete(succ),
                ];
            }
        }

        // When both branches of a br_if are the same, we can create a select
        // instruction to replace all of the differing arguments, and replace the
        // terminator with a jump into the target block.
        //
        // This hopefully unlocks further jump threading oportunities.
        if succ_count == 1 && all_dest_blocks_equal && terminator_opcode == Opcode::Brif {
            // Count how many arguments are different between the two block calls
            let true_args = branch_dests[0].args_slice(&self.func.dfg.value_lists);
            let false_args = branch_dests[1].args_slice(&self.func.dfg.value_lists);

            let differing_args_count = true_args
                .iter()
                .zip(false_args.iter())
                .filter(|(true_arg, false_arg)| true_arg != false_arg)
                .count() as u32;

            let select_cost = self.opcode_cost(Opcode::Select);
            let inline_cost = select_cost * differing_args_count;
            if inline_cost < self.config.max_inline_cost {
                return smallvec![JumpThreadAction::SelectifyBrIf(block)];
            }
        }

        // Try to inline our block into all of our predecessors. This clones all of
        // the instructions in this block into the previous blocks, and replaces
        // their block call with a block call into our successor.
        //
        // In a lot of cases, this will be free! For some reason we have a lot of
        // empty blocks with a single jump into another block. Those blocks
        // are a prime candidate for this transformation.
        //
        // However we are not limiting oursleves to empty blocks, we apply a cost
        // model to this block and inline if the total cost is below some treshold.
        if terminator_opcode == Opcode::Jump
            && pred_count >= 1
            && succ_count == 1
            && !is_self_succ
            && !self.block_has_side_effects(block)
            && !self.block_references_extern_values(block)
        {
            // We are going to duplicate this block, once per unique block call
            // from our predecessors. This is because different args into this
            // block will generate different instructions once inlined.
            let mut inline_calls: SmallVec<[_; 8]> = self
                .cfg
                .pred_iter(block)
                // Get all of the block calls in our predecessors
                .flat_map(|pred| {
                    self.func.dfg.insts[pred.inst]
                        .branch_destination(&self.func.dfg.jump_tables)
                        .into_iter()
                        .copied()
                        .map(move |call| (pred.block, call))
                })
                // Filter only for calls to our block
                .filter(|(_, call)| call.block(&self.func.dfg.value_lists) == block)
                .collect();

            // Deduplicate all of the calls, we shouldn't need to inline multiple
            // times for equivalent block calls, even if they appear at different
            // positions in the terminator.
            //
            // i.e. br_table v0, block0, [block1(v0), block1(v0), block1(v0)]
            // Only duplicates block1 once, since all args are the same.
            inline_calls.dedup_by(|(a_caller, a_call), (b_caller, b_call)| {
                let value_lists = &self.func.dfg.value_lists;
                a_caller == b_caller
                    && a_call.block(value_lists) == b_call.block(value_lists)
                    && a_call.args_slice(value_lists) == b_call.args_slice(value_lists)
            });
            debug_assert!(!inline_calls.is_empty());

            // Now we check if all of this duplication is still below the inline threshold
            let block_cost = self.block_cost(block);
            let inline_cost = block_cost * (inline_calls.len() as u32);
            if inline_cost <= self.config.max_inline_cost {
                let mut actions: SmallVec<_> = inline_calls
                    .into_iter()
                    .map(|(caller, call)| JumpThreadAction::InlineBlockCall {
                        caller_block: caller,
                        inlined_blockcall: call,
                    })
                    .collect();

                // After all this, we finally get to delete this block!
                actions.push(JumpThreadAction::Delete(block));

                return actions;
            }
        }

        smallvec![]
    }

    /// Calculates some cost for a given block. This cost model does not include
    /// terminators, since we'll usually want to evaluate the inlinable portion
    /// of the block (i.e. everything except the terminator.)
    fn block_cost(&self, block: Block) -> u32 {
        self.func
            .layout
            .block_insts(block)
            .map(|inst| self.func.dfg.insts[inst].opcode())
            .filter(|opcode| opcode.is_terminator())
            .map(|opcode| self.opcode_cost(opcode))
            .sum()
    }

    /// Calculates a made up cost for a given opcode.
    fn opcode_cost(&self, opcode: Opcode) -> u32 {
        match opcode {
            // Constants are pretty much free.
            Opcode::Iconst | Opcode::F32const | Opcode::F64const => 1,

            // Selects are somewhat expensive in this context, so we want to avoid too many
            // of them.
            Opcode::Select => 4,

            // Everything else is 2. We could do something more complex, but we can't predict
            // very well what is/is not going to get optimized out later.
            _ => 2,
        }
    }

    /// Does this block have any side effectful instructions?
    /// This analysis excludes the terminator instruction
    fn block_has_side_effects(&self, block: Block) -> bool {
        self.func
            .layout
            .block_insts(block)
            .filter(|inst| !self.func.dfg.insts[*inst].opcode().is_terminator())
            .any(|inst| has_side_effect(self.func, inst))
    }

    /// Does this block reference any values not defined in either the blockparams
    /// or it's own instructions.
    fn block_references_extern_values(&self, block: Block) -> bool {
        let block_params = self.func.dfg.block_params(block).into_iter().copied();
        let inst_values = self
            .func
            .layout
            .block_insts(block)
            .flat_map(|inst| self.func.dfg.inst_values(inst));

        block_params.chain(inst_values).any(|val| {
            let value_def = self.func.dfg.value_def(val);
            let arg_block = value_def.block();
            let inst_block = value_def
                .inst()
                .and_then(|def_inst| self.func.layout.inst_block(def_inst));

            // Defaulting to true here is the safe choice
            arg_block
                .or(inst_block)
                .map_or(true, |def_block| def_block != block)
        })
    }

    fn i64_from_iconst(&self, val: Value) -> Option<i64> {
        let dfg = &self.func.dfg;
        let inst = dfg.value_def(val).inst()?;
        let constant = match dfg.insts[inst] {
            InstructionData::UnaryImm {
                opcode: Opcode::Iconst,
                imm,
            } => imm.bits(),
            _ => return None,
        };
        let ty = dfg.value_type(dfg.first_result(inst));
        let shift_amt = std::cmp::max(0, 64 - ty.bits());
        Some((constant << shift_amt) >> shift_amt)
    }
}
