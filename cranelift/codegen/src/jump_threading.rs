//! Jump Thread analysis and optimization pass.

use alloc::vec::Vec;
use smallvec::{smallvec, SmallVec};

use crate::cursor::{Cursor, FuncCursor};
use crate::dominator_tree::DominatorTree;
use crate::flowgraph::{BlockPredecessor, ControlFlowGraph};
use crate::ir::{Block, BlockCall, Function, InstBuilder, Opcode};
use crate::loop_analysis::LoopAnalysis;
use crate::trace;
use core::fmt;

/// Represents a single action to be performed as part of the
/// jump thread analysis.
#[derive(Debug)]
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
                // TODO: This shouldn't be necessary in the future
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
                // jt.domtree.clear();
                // jt.domtree.compute(jt.func, jt.cfg);
                // jt.loop_analysis.clear();
            }
            JumpThreadAction::ReplaceWithJump(block, new_target) => {
                debug_assert_eq!(jt.cfg.succ_iter(block).count(), 1);
                let target_block = new_target.block(&jt.func.dfg.value_lists).clone();
                let target_values = new_target.args_slice(&jt.func.dfg.value_lists).to_vec();

                // Remove the terminator instruction on the block.
                let terminator = jt.func.layout.last_inst(block).unwrap();
                jt.func.layout.remove_inst(terminator);

                // Insert the new terminator as the last instruction
                // TODO: Copy srcloc
                let mut cursor = FuncCursor::new(jt.func).at_bottom(block);
                cursor.ins().jump(target_block, &target_values[..]);

                // We haven't changed the successor of this function, but we have changed
                // the terminator instruction so we need to recompute the cfg for this block
                jt.cfg.recompute_block(jt.func, block);
            }
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
    _loop_analysis: &'a mut LoopAnalysis,

    /// A queue of actions that we have pending
    actions: Vec<JumpThreadAction>,
}

impl<'a> JumpThreadingPass<'a> {
    pub fn new(
        func: &'a mut Function,
        cfg: &'a mut ControlFlowGraph,
        domtree: &'a mut DominatorTree,
        _loop_analysis: &'a mut LoopAnalysis,
    ) -> Self {
        // let mut domtree_preorder = DominatorTreePreorder::new();
        // domtree_preorder.compute(domtree, &func.layout);

        Self {
            func,
            cfg,
            domtree,
            // domtree_preorder,
            _loop_analysis,
            actions: Vec::new(),
        }
    }

    pub fn run(&mut self) {
        // Start by analyzing all blocks
        self.actions
            .extend(self.func.layout.blocks().map(JumpThreadAction::Analyze));

        // Run actions until we are done
        while let Some(action) = self.actions.pop() {
            action.run(self)
        }

        // Now that we're done rebuild whatever structures might be necessary
        // The CFG is always kept up to date, so we don't need to rebuild it here.

        self.domtree.clear();
        self.domtree.compute(self.func, self.cfg);
        // jt.loop_analysis.clear();
    }

    fn analyze_block(&mut self, block: Block) -> SmallVec<[JumpThreadAction; 1]> {
        // We assume that all blocks are reachable, and that unreachable blocks
        // were removed in previous passes.

        // If all of our terminator block calls are the same we can replace it with a jump terminator.
        // This can unlock further opportunities for optimization so we should reevaluate this block again.
        let terminator = self.func.layout.last_inst(block).unwrap();
        let branch_dests =
            self.func.dfg.insts[terminator].branch_destination(&self.func.dfg.jump_tables);
        let all_dests_equal = branch_dests.windows(2).all(|bc| {
            let (lhs, rhs) = (bc[0], bc[1]);

            let lhs_block = lhs.block(&self.func.dfg.value_lists);
            let lhs_args = lhs.args_slice(&self.func.dfg.value_lists);

            let rhs_block = rhs.block(&self.func.dfg.value_lists);
            let rhs_args = rhs.args_slice(&self.func.dfg.value_lists);

            lhs_block == rhs_block && lhs_args == rhs_args
        });
        if branch_dests.len() > 1 && all_dests_equal {
            return smallvec![
                JumpThreadAction::ReplaceWithJump(block, branch_dests[0]),
                // Reanalyze this block since the jump terminator may now
                // reveal better threading opportunities
                JumpThreadAction::Analyze(block),
            ];
        }

        // If we only have one successor, and that block only has one predecessor
        // we most definitley want to merge into it. We also check if the terminator
        // is a jump instruction, even with a singlesuccessor, we can still have
        // differing block args, which has to be handled differently.
        //
        // We also don't need to worry about computing the cost of this merger
        // since it can only reduce the total cost of the function.
        let succ_count = self.cfg.succ_iter(block).count();
        if succ_count == 1 {
            let succ = self.cfg.succ_iter(block).nth(0).unwrap();
            let succ_predecessors = self.cfg.pred_iter(succ).count();
            let terminator_opcode = self.func.dfg.insts[terminator].opcode();
            if succ_predecessors == 1 && terminator_opcode == Opcode::Jump {
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

        // TODO: Const evaluate br_table
        // TODO: Const evaluate br_if
        // TODO: Selectify br_if
        // TODO: Inline jump blocks

        smallvec![]
    }
}
