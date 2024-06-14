//! Jump Thread analysis and optimization pass.

use alloc::vec::Vec;
use smallvec::{smallvec, SmallVec};

use crate::cursor::{Cursor, FuncCursor};
use crate::dominator_tree::DominatorTree;
use crate::flowgraph::ControlFlowGraph;
use crate::ir::{Block, Function};
use crate::loop_analysis::LoopAnalysis;
use crate::settings::Flags;
use crate::trace;
use core::fmt;

/// Represents a single action to be performed as part of the
/// jump thread analysis.
#[derive(Debug, Clone, Copy)]
enum JumpThreadAction {
    /// This action skips this block and does not perform any transformations
    /// on it.
    Skip,

    /// Deletes this block from the function
    Delete(Block),
}

impl fmt::Display for JumpThreadAction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            JumpThreadAction::Skip => write!(f, "skip"),
            JumpThreadAction::Delete(block) => write!(f, "delete {block}"),
        }
    }
}

impl JumpThreadAction {
    fn run<'a>(self, jt: &mut JumpThreadingPass<'a>) {
        let mut cursor = FuncCursor::new(jt.func);

        match self {
            JumpThreadAction::Skip => {}
            JumpThreadAction::Delete(block) => {
                // Remove all instructions from `block`.
                while let Some(inst) = jt.func.layout.first_inst(block) {
                    trace!(" - {}", jt.func.dfg.display_inst(inst));
                    jt.func.layout.remove_inst(inst);
                }

                // Once the block is completely empty, we can update the CFG which removes it from any
                // predecessor lists.
                jt.cfg.recompute_block(jt.func, block);

                // Finally remove the block
                jt.func.layout.remove_block(block);
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
    domtree: &'a DominatorTree, //Preorder,
    /// Loop analysis results. We generally avoid performing actions
    /// on blocks belonging to loops since that can generate irreducible
    /// control flow
    loop_analysis: &'a LoopAnalysis,
    /// Compiler flags.
    flags: &'a Flags,
}

impl<'a> JumpThreadingPass<'a> {
    pub fn new(
        func: &'a mut Function,
        cfg: &'a mut ControlFlowGraph,
        domtree: &'a DominatorTree,
        loop_analysis: &'a LoopAnalysis,
        flags: &'a Flags,
    ) -> Self {
        // let mut domtree = DominatorTreePreorder::new();
        // domtree.compute(raw_domtree, &func.layout);

        Self {
            func,
            cfg,
            domtree,
            loop_analysis,
            flags,
        }
    }

    pub fn run(&mut self) {
        // let last_block = cursor.layout().last_block().unwrap();
        // let mut stack = vec![last_block];

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

            for action in actions.into_iter() {
                action.run(self)
            }
        }
    }

    fn analyze_block(&mut self, block: Block) -> SmallVec<[JumpThreadAction; 1]> {
        // If the block is unreachable, we can simply delete it and shouldn't
        // need to worry about it anymore
        if !self.domtree.is_reachable(block) {
            return smallvec![JumpThreadAction::Delete(block)];
        }

        smallvec![JumpThreadAction::Skip]
    }
}
