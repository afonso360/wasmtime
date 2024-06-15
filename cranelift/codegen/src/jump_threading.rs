//! Jump Thread analysis and optimization pass.

use alloc::vec::Vec;
use smallvec::{smallvec, SmallVec};

use crate::cursor::{Cursor, FuncCursor};
use crate::dominator_tree::DominatorTree;
use crate::flowgraph::{BlockPredecessor, ControlFlowGraph};
use crate::ir::{Block, Function, Opcode};
use crate::loop_analysis::LoopAnalysis;
use crate::trace;
use core::fmt;

/// Represents a single action to be performed as part of the
/// jump thread analysis.
#[derive(Debug)]
enum JumpThreadAction {
    /// This action skips this block and does not perform any transformations
    /// on it.
    Skip,

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
}

impl fmt::Display for JumpThreadAction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            JumpThreadAction::Skip => write!(f, "skip"),
            JumpThreadAction::Delete(block) => write!(f, "delete {block}"),
            JumpThreadAction::MergeIntoPredecessor {
                successor,
                predecessor,
            } => write!(f, "merge {successor} into {}", predecessor.block),
        }
    }
}

impl JumpThreadAction {
    fn run<'a>(self, jt: &mut JumpThreadingPass<'a>) {
        match self {
            JumpThreadAction::Skip => {}
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

                // // If there were any block parameters in block, then the last instruction in pred will
                // // fill these parameters. Make the block params aliases of the terminator arguments.
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
}

impl<'a> JumpThreadingPass<'a> {
    pub fn new(
        func: &'a mut Function,
        cfg: &'a mut ControlFlowGraph,
        domtree: &'a mut DominatorTree,
        loop_analysis: &'a mut LoopAnalysis,
    ) -> Self {
        // let mut domtree_preorder = DominatorTreePreorder::new();
        // domtree_preorder.compute(domtree, &func.layout);

        Self {
            func,
            cfg,
            domtree,
            // domtree_preorder,
            loop_analysis,
        }
    }

    pub fn run(&mut self) {
        // TODO: clean this up
        let blocks: Vec<_> = {
            let cursor = FuncCursor::new(self.func);
            let mut v: Vec<_> = cursor.layout().blocks().collect();
            v.reverse();
            v
        };
        for block in blocks {
            let actions = self.analyze_block(block);

            // Debug print the actions that we performed
            #[cfg(feature = "trace-log")]
            match actions.as_slice() {
                [single] => trace!("Evaluating {block}: {single}"),
                multi => {
                    trace!("Evaluating {block}:");
                    for action in multi {
                        trace!("\t- {action}");
                    }
                }
            };

            // Run all actions
            for action in actions.into_iter() {
                action.run(self)
            }
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

        // If we only have one predecessor, and that block only has one successor
        // we most definitley want to merge into it. We also check if the terminator
        // is a jump instruction, even with a since successor, we can still have differing
        // block args, which has to be handled differently.
        //
        // We also don't need to worry about computing the cost of this merger
        // since it can only reduce the total cost of the function.
        let pred_count = self.cfg.pred_iter(block).count();
        if pred_count == 1 {
            let pred = self.cfg.pred_iter(block).nth(0).unwrap();
            let pred_successors = self.cfg.succ_iter(pred.block).count();
            let pred_inst = pred.inst;
            let pred_opcode = self.func.dfg.insts[pred_inst].opcode();
            if pred_successors == 1 && pred_opcode == Opcode::Jump {
                return smallvec![
                    JumpThreadAction::MergeIntoPredecessor {
                        successor: block,
                        predecessor: pred,
                    },
                    JumpThreadAction::Delete(block),
                ];
            }
        }

        smallvec![JumpThreadAction::Skip]
    }
}
