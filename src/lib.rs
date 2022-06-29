use std::vec;

use petgraph::matrix_graph::{NotZero, UnMatrix};
#[derive(Debug, Clone, Copy)]
pub struct Point([f32; 2]);

#[derive(Clone, Copy)]
pub struct Circle {
    location: Point,
    radius: f32,
}

impl Circle {
    fn new(location: Point, radius: f32) -> Self {
        Self { location, radius }
    }
}
#[derive(Debug)]
pub struct Node {
    location: Point,
}
impl Node {
    fn new(location: Point) -> Self {
        Self { location }
    }
}

pub struct Edge {
    edge_type: EdgeType,
    weight: f32,
}

enum EdgeType {
    Surfing,
    Hugging,
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub fn build_graph(
    start: Node,
    end: Node,
    zones: Vec<Circle>,
) -> UnMatrix<Node, Edge, Option<Edge>> {
    let mut graph = UnMatrix::<Node, Edge, Option<Edge>>::with_capacity(32);
    graph.add_node(start);
    graph.add_node(end);
    return graph;
}

pub fn line_of_sight(node_1: Node, node_2: Node, zone: Circle) -> bool {
    /// Calculate u
    let c = zone.location.0;
    let a = node_1.location.0;
    let b = node_2.location.0;
    return false;
}

pub fn dot_product(p1: Vec<f32>, p2: Vec<f32>) -> f32 {
    let dot: f32 = p1.iter().zip(p2.iter()).map(|(a, b)| a * b).sum();
    return dot;
}

pub fn distance(p1: Vec<f32>, p2: Vec<f32>) -> f32 {
    let square_sum: f32 = p1
        .iter()
        .zip(p2.iter())
        .map(|(x1, x2)| (x2 - x1).powi(2))
        .sum();
    let distance = square_sum.sqrt();
    return distance;
}

pub fn add_pts(p1: Vec<f32>, p2: Vec<f32>) -> Vec<f32> {
    let point_sum: Vec<f32> = p1.iter().zip(p2.iter()).map(|(x1, x2)| x1 + x2).collect();
    return point_sum;
}

pub fn subtrac_pts(p1: Vec<f32>, p2: Vec<f32>) -> Vec<f32> {
    let difference: Vec<f32> = p1.iter().zip(p2.iter()).map(|(x1, x2)| x1 - x2).collect();
    return difference;
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
        let start = Node::new(Point([0.0, 0.0]));
        let end = Node::new(Point([5.0, 5.0]));
        let circle = Circle::new(Point([2.0, 2.0]), 2.0);
        let circle_vec = vec![circle];
        let graph = build_graph(start, end, circle_vec);
        assert_eq!(graph.node_count(), 2);
        // print!("{:?}", graph)
    }
}
