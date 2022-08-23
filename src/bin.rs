use avoidrs::{Circle, Graph, Node};
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;

fn main() {
    let start = Node::new([0.0, 0.0]);
    let end = Node::new([5.0, 5.0]);
    let circle = Circle::new([2.0, 2.0], dec!(2.0));
    let circle_vec = vec![circle];
    let graph = Graph::build_graph(start, end, circle_vec);
    let nodes = graph.neighbors(start);
}
