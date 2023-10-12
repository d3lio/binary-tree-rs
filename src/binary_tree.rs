#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(transparent)]
// TODO Check if this token comes from the same tree instance which prevents invalid indices.
pub struct Token(usize);

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

    fn with_parent(parent: Token, data: T) -> Self {
        Self {
            data,
            parent: Some(parent),
            children: (None, None),
        }
    }
}

pub struct BinaryTree<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    arena: Vec<Option<Node<T>>>,
    root: Option<Token>,
}

impl<T> Default for BinaryTree<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    fn default() -> Self {
        Self {
            arena: Default::default(),
            root: Default::default(),
        }
    }
}

impl<T> BinaryTree<T>
where T: Eq + PartialEq + Ord + PartialOrd {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, token: Token) -> &T {
        &self.arena[token.0].as_ref().unwrap().data
    }

    pub fn get_mut(&mut self, token: Token) -> &mut T {
        &mut self.arena[token.0].as_mut().unwrap().data
    }

    fn get_node(&self, token: Token) -> &Node<T> {
        self.arena[token.0].as_ref().unwrap()
    }

    fn get_node_mut(&mut self, token: Token) -> &mut Node<T> {
        self.arena[token.0].as_mut().unwrap()
    }

    pub fn add(&mut self, data: T) -> Token {
        let mut current = self.root;
        while let Some(token) = current {
            if self.get(token) >= &data {
                if let Some(left) = self.get_node(token).children.0 {
                    current = Some(left);
                    continue;
                }
                return self.add_left(token, data);
            } else if self.get(token) < &data {
                if let Some(right) = self.get_node(token).children.1 {
                    current = Some(right);
                    continue;
                }
                return self.add_right(token, data);
            }
        }
        self.add_root(data)
    }

    pub fn remove(&mut self, token: Token) -> T {
        let node = self.arena.remove(token.0).unwrap();

        todo!();

        node.data
    }

    fn add_root(&mut self, data: T) -> Token {
        assert!(self.arena.is_empty());

        self.arena.push(Some(Node::new(data)));
        let token = Token(0);
        self.root = Some(token);
        token
    }

    fn add_left(&mut self, parent: Token, data: T) -> Token {
        self.arena.push(Some(Node::with_parent(parent, data)));
        let ret = Token(self.arena.len() - 1);
        self.get_node_mut(parent).children.0 = Some(ret);
        ret
    }

    fn add_right(&mut self, parent: Token, data: T) -> Token {
        self.arena.push(Some(Node::with_parent(parent, data)));
        let ret = Token(self.arena.len() - 1);
        self.get_node_mut(parent).children.1 = Some(ret);
        ret
    }

    pub fn iter(&self) -> Iter<T> {
        Iter::new(self)
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


struct Iter<'a, T>
where T: Eq + PartialEq + Ord + PartialOrd {
    tree: &'a BinaryTree<T>,
    stack: Vec<Token>,
}

impl<'a, T> Iter<'a, T>
where T: Eq + PartialEq + Ord + PartialOrd  {
    fn new(tree: &'a BinaryTree<T>) -> Self {
        let mut stack = vec![];
        Self::push_all_left_children(tree, tree.root, &mut stack);

        Self {
            tree,
            stack,
        }
    }

    fn push_all_left_children(tree: &'a BinaryTree<T>, mut current: Option<Token>, stack: &mut Vec<Token>) {
        while let Some(token) = current.take() {
            stack.push(token);
            if let (Some(left), _) = tree.get_node(token).children {
                current = Some(left);
            }
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T>
where T: Eq + PartialEq + Ord + PartialOrd  {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let top = self.stack.pop();

        Self::push_all_left_children(self.tree, top.and_then(|top| self.tree.get_node(top).children.1), &mut self.stack);

        top.map(|token| self.tree.get(token))
    }

    // Nightly
    // fn is_sorted(self) -> bool {
    //     true
    // }
}

#[cfg(test)]
mod test {
    use super::*;

    fn debug<T: Eq + PartialEq + Ord + PartialOrd>(bt: &BinaryTree<T>) {
        println!("{:?}", bt.arena.iter()
            .filter_map(|item| item.as_ref())
            .map(|item| item.children)
            .collect::<Vec<_>>());
    }

    macro_rules! assert_iter_eq {
        ($bt:expr, $expected:expr) => {
            assert_eq!($bt.iter().collect::<Vec<_>>(), $expected.iter().collect::<Vec<_>>());
        };
    }

    #[test]
    fn empty() {
        let bt = BinaryTree::<u32>::new();

        assert_eq!(bt.arena.len(), 0);
        assert_eq!(bt.root, None);
        assert_eq!(bt.iter().collect::<Vec<_>>(), Vec::<&u32>::new());
    }

    #[test]
    fn one_element() {
        let mut bt = BinaryTree::<u32>::new();
        let token = bt.add(1);

        assert_eq!(token.0, 0);
        assert_eq!(bt.arena.len(), 1);
        assert_eq!(bt.root, Some(Token(0)));
        assert_iter_eq!(bt, vec![1]);
    }

    #[test]
    fn multiple_elements() {
        let mut bt = BinaryTree::<u32>::new();
        let token1 = bt.add(1);
        let token2 = bt.add(5);
        let token3 = bt.add(2);
        let token4 = bt.add(3);

        assert_eq!(token1.0, 0);
        assert_eq!(token2.0, 1);
        assert_eq!(token3.0, 2);
        assert_eq!(token4.0, 3);
        assert_iter_eq!(bt, vec![1, 2, 3, 5]);
    }

    #[test]
    fn complex_tree() {
        let input = vec![7, 2, 1, 5, 10, 3, 4, 6, 8, 9];
        let bt = input.into_iter().collect::<BinaryTree<_>>();

        assert_iter_eq!(bt, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
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

        assert_iter_eq!(bt, vec![1, 2, 4, 5, 6, 7, 8, 9, 10]);
    }
}
