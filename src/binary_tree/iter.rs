use crate::{BinaryTree, Token};

/// Iterates a [`BinaryTree`].
///
/// Guaranteed to be sorted since this is a binary search tree.
///
/// ```
/// # use binary_tree::*;
/// # fn main() {
/// let bt = vec![5, 2, 10, 7, 1].into_iter().collect::<BinaryTree<_>>();
/// let vec = bt.iter().collect::<Vec<_>>();
/// assert_eq!(vec, vec![&1, &2, &5, &7, &10]);
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Iter<'a, T>
where T: Eq + PartialEq + Ord + PartialOrd {
    tree: &'a BinaryTree<T>,
    stack: Vec<Token>,

}

impl<'a, T> Iter<'a, T>
where T: Eq + PartialEq + Ord + PartialOrd {
    pub(crate) fn new(tree: &'a BinaryTree<T>) -> Self {
        let mut inst = Self {
            tree,
            stack: vec![],
        };

        inst.push_all_left_children(tree.root);

        inst
    }

    fn push_all_left_children(&mut self, mut current: Option<Token>) {
        while let Some(token) = current.take() {
            self.stack.push(token);
            if let (Some(left), _) = self.tree.node(token).children {
                current = Some(left);
            }
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T>
where T: Eq + PartialEq + Ord + PartialOrd {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let top = self.stack.pop();

        self.push_all_left_children(top.and_then(|top| self.tree.node(top).children.1));

        top.map(|token| &self.tree[token])
    }

    fn count(self) -> usize {
        self.tree.size()
    }

    #[cfg(is_sorted)]
    fn is_sorted(self) -> bool {
        true
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.tree.size(), Some(self.tree.size()))
    }

    fn min(self) -> Option<Self::Item> {
        self.tree.min()
    }

    fn max(self) -> Option<Self::Item> {
        self.tree.max()
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T>
where T: Eq + PartialEq + Ord + PartialOrd {
    fn len(&self) -> usize {
        self.tree.size()
    }

    #[cfg(exact_size_is_empty)]
    fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }
}

#[cfg(test)]
mod test {
    use crate::BinaryTree;

    #[test]
    fn min() {
        let bt = vec![1, 5, 2, 3].into_iter().collect::<BinaryTree<_ >>();
        assert_eq!(bt.iter().min(), Some(&1));
    }

    #[test]
    fn min_none() {
        let bt = BinaryTree::<u32>::new();
        assert_eq!(bt.iter().min(), None);
    }

    #[test]
    fn max() {
        let bt = vec![1, 5, 2, 3].into_iter().collect::<BinaryTree<_ >>();
        assert_eq!(bt.iter().max(), Some(&5));
    }

    #[test]
    fn max_none() {
        let bt = BinaryTree::<u32>::new();
        assert_eq!(bt.iter().max(), None);
    }
}
