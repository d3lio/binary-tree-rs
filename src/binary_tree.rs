use std::{num::NonZeroUsize, ops::Index};

/// Index for values inside a [`BinaryTree`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct Token(NonZeroUsize);

impl Token {
    fn new(index: usize) -> Self {
        Self(NonZeroUsize::new(index).unwrap())
    }
}

#[derive(Debug)]
struct Node<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    data: T,
    parent: Option<Token>,
    children: (Option<Token>, Option<Token>),
}

impl<T> Node<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    fn new(data: T) -> Self {
        Self {
            data,
            parent: None,
            children: (None, None),
        }
    }
}

/// Binary search tree.
///
/// This structure is unbalanced.
///
/// ```
/// # use binary_tree::*;
/// # fn main() {
/// let mut bt = BinaryTree::new();
/// let token = bt.add(1);
/// assert_eq!(bt.get(token), Some(&1));
/// # }
/// ```
#[derive(Debug)]
pub struct BinaryTree<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    arena: Vec<Option<Node<T>>>,
    root: Option<Token>,
    size: usize,
}

impl<T> Default for BinaryTree<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    fn default() -> Self {
        Self {
            arena: vec![None],
            root: Default::default(),
            size: Default::default(),
        }
    }
}

impl<T> BinaryTree<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get reference to the data indexed by `token`.
    ///
    /// ```
    /// # use binary_tree::*;
    /// # fn main() {
    /// let mut bt = BinaryTree::new();
    /// let token = bt.add(1);
    /// assert_eq!(bt.get(token), Some(&1));
    /// # }
    /// ```
    pub fn get(&self, token: Token) -> Option<&T> {
        self.arena[token.0.get()].as_ref().map(|node| &node.data)
    }

    /// Add a new item to the tree.
    ///
    /// ```
    /// # use binary_tree::*;
    /// # fn main() {
    /// let mut bt = BinaryTree::new();
    /// let token = bt.add(1);
    /// assert_eq!(bt.get(token), Some(&1));
    /// # }
    /// ```
    pub fn add(&mut self, data: T) -> Token {
        let token = self.create_node(data);
        self.add_by_token(self.root, token);
        token
    }

    fn add_by_token(&mut self, parent: Option<Token>, token: Token) {
        let data = self.get(token).unwrap();

        match parent {
            None => self.root = Some(token),
            Some(mut current) => {
                loop {
                    if self.get(current).unwrap() >= data {
                        if let Some(left) = self.node(current).children.0 {
                            current = left;
                        } else {
                            return self.insert_left(current, token);
                        }
                    } else {
                        if let Some(right) = self.node(current).children.1 {
                            current = right;
                        } else {
                            return self.insert_right(current, token);
                        }
                    }
                }
            },
        }
    }

    /// Remove an item from the tree.
    ///
    /// ```
    /// # use binary_tree::*;
    /// # fn main() {
    /// let mut bt = BinaryTree::new();
    /// let token = bt.add(1);
    /// assert_eq!(bt.size(), 1);
    /// bt.remove(token);
    /// assert_eq!(bt.size(), 0);
    /// # }
    /// ```
    pub fn remove(&mut self, token: Token) -> Option<T> {
        self.size -= 1;

        let node = self.arena[token.0.get()].take()?;

        match node.parent {
            None => {
                match node.children {
                    (None, None) => {
                        self.clear();
                    },
                    (Some(left), None) => {
                        self.root = Some(left);
                        self.node_mut(left).parent = None;
                    }
                    (None, Some(right)) => {
                        self.root = Some(right);
                        self.node_mut(right).parent = None;
                    },
                    (Some(left), Some(right)) => {
                        self.root = Some(right);
                        self.node_mut(left).parent = None;
                        self.node_mut(right).parent = None;
                        self.add_by_token(Some(right), left);
                    },
                }
            },
            Some(parent) => {
                match node.children {
                    (None, None) => {
                        let parent_node = self.node_mut(parent);
                        if parent_node.children.0 == Some(token) {
                            parent_node.children.0 = None
                        } else {
                            parent_node.children.1 = None
                        }
                    },
                    (Some(left), None) => {
                        let parent_node = self.node_mut(parent);
                        if parent_node.children.0 == Some(token) {
                            parent_node.children.0 = Some(left)
                        } else {
                            parent_node.children.1 = Some(left)
                        }
                        self.node_mut(left).parent = Some(parent);
                    }
                    (None, Some(right)) => {
                        let parent_node = self.node_mut(parent);
                        if parent_node.children.0 == Some(token) {
                            parent_node.children.0 = Some(right)
                        } else {
                            parent_node.children.1 = Some(right)
                        }
                        self.node_mut(right).parent = Some(parent);
                    },
                    (Some(left), Some(right)) => {
                        let parent_node = self.node_mut(parent);
                        if parent_node.children.0 == Some(token) {
                            parent_node.children.0 = Some(right);
                        } else {
                            parent_node.children.1 = Some(right);
                        }
                        self.node_mut(right).parent = Some(parent);
                        self.add_by_token(Some(right), left);
                    },
                }
            },
        };

        while let Some(None) = self.arena[1..].last() {
            self.arena.pop();
        }

        Some(node.data)
    }

    /// Get the size of the tree.
    ///
    /// ```
    /// # use binary_tree::*;
    /// # fn main() {
    /// let mut bt = BinaryTree::new();
    /// let token = bt.add(1);
    /// assert_eq!(bt.size(), 1);
    /// bt.remove(token);
    /// assert_eq!(bt.size(), 0);
    /// # }
    /// ```
    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if the tree is empty.
    ///
    /// ```
    /// # use binary_tree::*;
    /// # fn main() {
    /// let mut bt = BinaryTree::new();
    /// assert_eq!(bt.is_empty(), true);
    /// let token = bt.add(1);
    /// assert_eq!(bt.is_empty(), false);
    /// bt.remove(token);
    /// assert_eq!(bt.is_empty(), true);
    /// # }
    /// ```
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Clear the tree.
    ///
    /// ```
    /// # use binary_tree::*;
    /// # fn main() {
    /// let mut bt = BinaryTree::new();
    /// let token = bt.add(1);
    /// assert_eq!(bt.size(), 1);
    /// bt.clear();
    /// assert_eq!(bt.size(), 0);
    /// # }
    /// ```
    pub fn clear(&mut self) {
        self.arena = vec![None];
        self.root = None;
        self.size = 0;
    }

    fn node(&self, token: Token) -> &Node<T> {
        self.arena[token.0.get()].as_ref().unwrap()
    }

    fn node_mut(&mut self, token: Token) -> &mut Node<T> {
        self.arena[token.0.get()].as_mut().unwrap()
    }

    fn create_node(&mut self, data: T) -> Token {
        self.size += 1;

        let node = Some(Node::new(data));
        let free_index = self.arena.iter()
            .enumerate()
            .skip(1)
            .find_map(|(index, item)|  match item.as_ref() {
                None => Some(index),
                Some(_) => None,
            });

        if let Some(index) = free_index {
            self.arena[index] = node;
            Token::new(index)
        } else {
            self.arena.push(node);
            Token::new(self.arena.len() - 1)
        }
    }

    fn insert_left(&mut self, parent: Token, token: Token) {
        self.node_mut(parent).children.0 = Some(token);
        self.node_mut(token).parent = Some(parent);
    }

    fn insert_right(&mut self, parent: Token, token: Token) {
        self.node_mut(parent).children.1 = Some(token);
        self.node_mut(token).parent = Some(parent);
    }

    /// See [`Iter`].
    pub fn iter(&self) -> Iter<T> {
        Iter::new(self)
    }
}

impl<T> Index<Token> for BinaryTree<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    type Output = T;

    fn index(&self, index: Token) -> &Self::Output {
        &self.arena[index.0.get()].as_ref().unwrap().data
    }
}

impl<T> FromIterator<T> for BinaryTree<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut bt = BinaryTree::new();
        for node in iter {
            bt.add(node);
        }
        bt
    }
}

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
pub struct Iter<'a, T>
where T: Eq + PartialEq + Ord + PartialOrd {
    tree: &'a BinaryTree<T>,
    stack: Vec<Token>,
}

impl<'a, T> Iter<'a, T>
where T: Eq + PartialEq + Ord + PartialOrd {
    fn new(tree: &'a BinaryTree<T>) -> Self {
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

    // Nightly
    // fn is_sorted(self) -> bool {
    //     true
    // }
}

impl<T> IntoIterator for BinaryTree<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<'a, T> IntoIterator for &'a BinaryTree<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    type Item = &'a T;

    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

/// Iterates a [`BinaryTree`].
///
/// Guaranteed to be sorted since this is a binary search tree.
///
/// ```
/// # use binary_tree::*;
/// # fn main() {
/// let bt = vec![5, 2, 10, 7, 1].into_iter().collect::<BinaryTree<_>>();
/// let vec = bt.into_iter().collect::<Vec<_>>();
/// assert_eq!(vec, vec![1, 2, 5, 7, 10]);
/// # }
/// ```
pub struct IntoIter<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    tree: BinaryTree<T>,
    stack: Vec<Token>,
}

impl<T> IntoIter<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    fn new(tree: BinaryTree<T>) -> Self {
        let mut stack = vec![];
        Self::push_all_left_children(&tree, tree.root, &mut stack);

        Self {
            tree,
            stack,
        }
    }

    fn push_all_left_children(tree: &BinaryTree<T>, mut current: Option<Token>, stack: &mut Vec<Token>) {
        while let Some(token) = current.take() {
            stack.push(token);
            if let (Some(left), _) = tree.node(token).children {
                current = Some(left);
            }
        }
    }
}

impl<T> Iterator for IntoIter<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let top = self.stack.pop();

        Self::push_all_left_children(&self.tree, top.and_then(|top| self.tree.node(top).children.1), &mut self.stack);

        top.and_then(|token| self.tree.arena[token.0.get()].take().map(|node| node.data))
    }

    // Nightly
    // fn is_sorted(self) -> bool {
    //     true
    // }
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! assert_iter_eq {
        ($bt:expr, $expected:expr) => {
            assert_eq!($bt.iter().collect::<Vec<_>>(), $expected.iter().collect::<Vec<_>>());
        };
    }

    #[test]
    fn empty() {
        let bt = BinaryTree::<u32>::new();

        assert_eq!(bt.arena.len(), 1);
        assert_eq!(bt.root, None);
        assert_eq!(bt.size(), 0);
        assert_eq!(bt.iter().collect::<Vec<_>>(), Vec::<&u32>::new());
    }

    #[test]
    fn one_element() {
        let mut bt = BinaryTree::<u32>::new();
        let token = bt.add(1);

        assert_eq!(token.0.get(), 1);
        assert_eq!(bt.arena.len(), 2);
        assert_eq!(bt.size(), 1);
        assert_eq!(bt.root, Some(Token::new(1)));
        assert_iter_eq!(bt, vec![1]);
    }

    #[test]
    fn multiple_elements() {
        let mut bt = BinaryTree::<u32>::new();
        assert_eq!(bt.size(), 0);
        let token1 = bt.add(1);
        let token2 = bt.add(5);
        let token3 = bt.add(2);
        let token4 = bt.add(3);
        assert_eq!(bt.size(), 4);

        assert_eq!(token1.0.get(), 1);
        assert_eq!(token2.0.get(), 2);
        assert_eq!(token3.0.get(), 3);
        assert_eq!(token4.0.get(), 4);
        assert_iter_eq!(bt, vec![1, 2, 3, 5]);
    }

    #[test]
    fn clear() {
        let mut bt = vec![1, 5, 2, 3].into_iter().collect::<BinaryTree<_>>();

        assert_eq!(bt.arena.len(), 5);
        assert_eq!(bt.size(), 4);
        assert_eq!(bt.is_empty(), false);

        bt.clear();

        assert_eq!(bt.arena.len(), 1);
        assert_eq!(bt.root, None);
        assert_eq!(bt.size(), 0);
        assert_eq!(bt.is_empty(), true);
    }
}

#[cfg(test)]
mod remove_test {
    use super::*;

    macro_rules! assert_iter_eq {
        ($bt:expr, $expected:expr) => {
            assert_eq!($bt.iter().collect::<Vec<_>>(), $expected.iter().collect::<Vec<_>>());
        };
    }

    macro_rules! assert_into_iter_eq {
        ($bt:expr, $expected:expr) => {
            assert_eq!($bt.into_iter().collect::<Vec<_>>(), $expected.into_iter().collect::<Vec<_>>());
        };
    }


    #[test]
    fn complex_tree() {
        let input = vec![7, 2, 1, 5, 10, 3, 4, 6, 8, 9];
        let bt = input.into_iter().collect::<BinaryTree<_>>();

        assert_eq!(bt.size(), 10);
        assert_iter_eq!(bt, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_into_iter_eq!(bt, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn remove_left_with_right_child() {
        let mut bt = BinaryTree::<u32>::new();
        bt.add(7);
        bt.add(2);
        bt.add(1);
        bt.add(5);
        bt.add(10);
        let token = bt.add(3);
        bt.add(4);
        bt.add(6);
        bt.add(8);
        bt.add(9);

        bt.remove(token);

        assert_eq!(bt.size(), 9);
        assert_iter_eq!(bt, vec![1, 2, 4, 5, 6, 7, 8, 9, 10]);
        assert_into_iter_eq!(bt, vec![1, 2, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn remove_left_with_left_child() {
        let mut bt = BinaryTree::<u32>::new();
        bt.add(7);
        bt.add(2);
        let token = bt.add(1);
        bt.add(5);
        bt.add(10);
        bt.add(3);
        bt.add(4);
        bt.add(6);
        bt.add(8);
        bt.add(9);

        bt.remove(token);

        assert_eq!(bt.size(), 9);
        assert_iter_eq!(bt, vec![2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_into_iter_eq!(bt, vec![2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn remove_left_with_two_children() {
        let mut bt = BinaryTree::<u32>::new();
        bt.add(7);
        let token = bt.add(2);
        bt.add(1);
        bt.add(3);
        bt.add(4);

        bt.remove(token);

        assert_eq!(bt.size(), 4);
        assert_iter_eq!(bt, vec![1, 3, 4, 7]);
        assert_into_iter_eq!(bt, vec![1, 3, 4, 7]);
    }

    #[test]
    fn remove_left_with_no_children() {
        let mut bt = BinaryTree::<u32>::new();
        bt.add(7);
        bt.add(2);
        let token = bt.add(1);
        bt.add(5);
        bt.add(10);
        bt.add(3);
        bt.add(4);
        bt.add(6);
        bt.add(8);
        bt.add(9);

        bt.remove(token);

        assert_eq!(bt.size(), 9);
        assert_iter_eq!(bt, vec![2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_into_iter_eq!(bt, vec![2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn remove_right_with_right_child() {
        let mut bt = BinaryTree::<u32>::new();
        bt.add(7);
        bt.add(2);
        bt.add(1);
        bt.add(5);
        bt.add(10);
        bt.add(3);
        bt.add(4);
        bt.add(6);
        let token = bt.add(8);
        bt.add(9);

        bt.remove(token);

        assert_eq!(bt.size(), 9);
        assert_iter_eq!(bt, vec![1, 2, 3, 4, 5, 6, 7, 9, 10]);
        assert_into_iter_eq!(bt, vec![1, 2, 3, 4, 5, 6, 7, 9, 10]);
    }

    #[test]
    fn remove_right_with_left_child() {
        let mut bt = BinaryTree::<u32>::new();
        bt.add(7);
        bt.add(2);
        bt.add(1);
        bt.add(5);
        let token = bt.add(10);
        bt.add(3);
        bt.add(4);
        bt.add(6);
        bt.add(8);

        bt.remove(token);

        assert_eq!(bt.size(), 8);
        assert_iter_eq!(bt, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_into_iter_eq!(bt, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn remove_right_with_two_children() {
        let mut bt = BinaryTree::<u32>::new();
        bt.add(7);
        let token = bt.add(10);
        bt.add(9);
        bt.add(11);
        bt.add(8);
        bt.add(12);

        bt.remove(token);

        assert_eq!(bt.size(), 5);
        assert_iter_eq!(bt, vec![7, 8, 9, 11, 12]);
        assert_into_iter_eq!(bt, vec![7, 8, 9, 11, 12]);
    }

    #[test]
    fn remove_right_with_no_children() {
        let mut bt = BinaryTree::<u32>::new();
        bt.add(7);
        bt.add(2);
        bt.add(1);
        bt.add(5);
        bt.add(10);
        bt.add(3);
        bt.add(4);
        bt.add(6);
        bt.add(8);
        let token = bt.add(9);

        bt.remove(token);

        assert_eq!(bt.size(), 9);
        assert_iter_eq!(bt, vec![1, 2, 3, 4, 5, 6, 7, 8, 10]);
        assert_into_iter_eq!(bt, vec![1, 2, 3, 4, 5, 6, 7, 8, 10]);
    }

    #[test]
    fn remove_half() {
        let mut bt = BinaryTree::<u32>::new();
        let tokens = [
            bt.add(7),
            bt.add(2),
            bt.add(1),
            bt.add(5),
            bt.add(10),
            bt.add(3),
            bt.add(4),
            bt.add(6),
            bt.add(8),
            bt.add(9),
        ];

        for token in &tokens[0..5] {
            bt.remove(*token);
        }

        assert_eq!(bt.size(), 5);
        assert_iter_eq!(bt, vec![3, 4, 6, 8, 9]);
        assert_into_iter_eq!(bt, vec![3, 4, 6, 8, 9]);
    }

    #[test]
    fn remove_when_duplicate() {
        let mut bt = BinaryTree::<u32>::new();
        let token = bt.add(1);
        bt.add(1);
        bt.add(1);

        bt.remove(token);

        assert_eq!(bt.size(), 2);
        assert_iter_eq!(bt, vec![1, 1]);
        assert_into_iter_eq!(bt, vec![1, 1]);
    }

    #[test]
    fn into_iterator_for_loop_consume() {
        let bt = vec![1, 5, 2, 3].into_iter().collect::<BinaryTree<_>>();
        let mut collected = vec![];
        for item in bt {
            collected.push(item);
        }

        assert_eq!(collected, vec![1, 2, 3, 5]);
    }

    #[test]
    fn into_iterator_for_loop_ref() {
        let bt = vec![1, 5, 2, 3].into_iter().collect::<BinaryTree<_>>();
        let mut collected = vec![];
        for item in &bt {
            collected.push(item);
        }

        assert_eq!(collected, vec![&1, &2, &3, &5]);
    }

    #[test]
    fn reuse_free_arena_space() {
        let mut bt = BinaryTree::<u32>::new();
        let mut tokens = vec![
            bt.add(2),
            bt.add(1),
            bt.add(3),
        ];

        assert_eq!(bt.size(), 3);
        assert_eq!(bt.arena.len(), 4);

        bt.remove(tokens[0]);
        bt.remove(tokens[1]);

        assert_eq!(bt.size(), 1);
        assert_eq!(bt.arena.len(), 4);

        tokens.push(bt.add(2));

        assert_eq!(bt.size(), 2);
        assert_eq!(bt.arena.len(), 4);

        tokens.push(bt.add(3));

        assert_eq!(bt.size(), 3);
        assert_eq!(bt.arena.len(), 4);

        tokens.push(bt.add(5));

        assert_eq!(bt.size(), 4);
        assert_eq!(bt.arena.len(), 5);
    }

    #[test]
    fn arena_remove_last_if_none_optimization() {
        let mut bt = BinaryTree::<u32>::new();
        let mut tokens = vec![
            bt.add(2),
            bt.add(1),
            bt.add(3),
        ];

        assert_eq!(bt.size(), 3);
        assert_eq!(bt.arena.len(), 4);

        bt.remove(tokens[1]);
        bt.remove(tokens[2]);

        assert_eq!(bt.size(), 1);
        assert_eq!(bt.arena.len(), 2);

        tokens.push(bt.add(2));

        assert_eq!(bt.size(), 2);
        assert_eq!(bt.arena.len(), 3);

        tokens.push(bt.add(3));

        assert_eq!(bt.size(), 3);
        assert_eq!(bt.arena.len(), 4);

        tokens.push(bt.add(5));

        assert_eq!(bt.size(), 4);
        assert_eq!(bt.arena.len(), 5);
    }

    #[test]
    fn remove_all_elements() {
        let mut bt = BinaryTree::<u32>::new();
        let tokens = vec![
            bt.add(2),
            bt.add(1),
            bt.add(3),
        ];

        assert_eq!(bt.size(), 3);
        assert_eq!(bt.arena.len(), 4);

        for token in tokens {
            bt.remove(token);
        }

        assert_eq!(bt.size(), 0);
        assert_eq!(bt.arena.len(), 1);
    }

    #[test]
    fn remove_all_elements_and_add_more() {
        let mut bt = BinaryTree::<u32>::new();
        let tokens = vec![
            bt.add(2),
            bt.add(1),
            bt.add(3),
        ];

        for token in tokens {
            bt.remove(token);
        }

        bt.add(5);
        bt.add(4);
        bt.add(7);
        bt.add(6);

        assert_eq!(bt.size(), 4);
        assert_iter_eq!(bt, vec![4, 5, 6, 7]);
        assert_into_iter_eq!(bt, vec![4, 5, 6, 7]);
    }
}
