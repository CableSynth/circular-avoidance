use avoidrs::{a_star, reconstruct_path, Graph, Node, Zone};
use itertools::Itertools;
use serde_json::Result;
use std::{
    fs::{File, OpenOptions},
    vec,
};

fn main() {
    let zone_list = [([2.0, 2.0], 2.0), ([9.0, 9.0], 3.0), ([9.0, 5.0], 3.0)];
    let start = Node::new([0.0, 0.0], None);
    let end = Node::new([30.0, 30.0], None);
    //  let z = Zone::new([2.0, 2.0], 2.0);
    //  let z1 = Zone::new([9.0, 9.0], 3.0);
    //  let z2 = Zone::new([15.0, 15.0], 6.0);
    let zs = zone_list.iter().map(|z| Zone::new(z.0, z.1)).collect_vec();
    // let circle_vec = vec![z, z1, z2];
    let mut graph = Graph::build_graph(start, end, zs);
    let (came_from, _) = a_star(&mut graph);
    let f = File::create("json_out/graph.json").expect("Failed to create");
    let j = serde_json::to_writer_pretty(f, &graph).expect("can't write");
    let found_path = reconstruct_path(came_from, start, end);
    let output_str = serde_json::to_string(&found_path);
    let f = File::create("json_out/found_path.json").expect("Failed to create");
    let j = serde_json::to_writer_pretty(f, &found_path).expect("can't write");
    println!("{:?}", output_str);
}
