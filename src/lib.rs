use std::vec;

use petgraph::{matrix_graph::{UnMatrix, NotZero}};
#[derive(Clone, Copy)]
pub struct Circle {
    location: (f64, f64),
    radius: f32
}

impl Circle {
    fn new (
        start: f64,
        end: f64,
        radius: f32
    ) -> Self {
        Self { 
            location: (start, end), 
            radius: radius
        }
    }
}
#[derive(Debug)]
pub struct Node {
    location: (f64, f64),
}
impl Node {
    fn new(
        start: f64,
        end: f64,
    ) -> Self{
        Self { location: (start, end) }
    }
}

pub struct Edge {
    edge_type: EdgeType,
    weight: f64
}

enum EdgeType {
    Surfing,
    Hugging,
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub fn build_graph(start: Node, end: Node, zones: Vec<Circle>) -> UnMatrix<Node, Edge, Option<Edge>>{
    let mut graph = UnMatrix::<Node, Edge, Option<Edge>>::with_capacity(32);
    graph.add_node(start);
    graph.add_node(end);
    return graph;
}

pub fn line_of_sight() {

}

#[cfg(test)]
mod tests {
    use petgraph::visit::IntoNodeReferences;

    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn simple_graph() {
        let start = Node::new(0.0, 0.0);
        let end = Node::new(5.0, 5.0);
        let circle = Circle::new(2.0, 2.0, 1.0);
        let circle_vec = vec![circle];
        let graph = build_graph(start, end, circle_vec);
        assert_eq!(graph.node_count(), 2);
        // print!("{:?}", graph)
    }
}
