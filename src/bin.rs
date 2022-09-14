use avoidrs::{a_star, reconstruct_path, Graph, Node, Zone};
use serde_json::Result;
use std::{
    fs::{File, OpenOptions},
    vec,
};

fn main() {
    let start = Node::new([0.0, 0.0], None);
    let end = Node::new([30.0, 30.0], None);
    let z = Zone::new([2.0, 2.0], 2.0);
    let circle_vec = vec![z];
    let mut graph = Graph::build_graph(start, end, circle_vec);
    let (came_from, _) = a_star(&mut graph);
    let f = File::create("json_out/test.json").expect("Failed to create");
    let j = serde_json::to_writer_pretty(f, &graph).expect("can't write");
    // let path = reconstruct_path(came_from, start, end);
}
