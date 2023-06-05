use avoidrs::{a_star, reconstruct_path, Graph, Node, Zone};
use clap::Parser;
use itertools::Itertools;
use std::fs::File;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input CSV file
    #[arg(short, long)]
    file_name: String,
}

fn main() {
    //let args = Args::parse();
    let zone_list = [([9.0, 5.0], 3.0), ([15.0, 15.0], 6.0),
                                           ([15.0, 5.0], 6.0), ([15.0, 25.0], 6.0),
                                           ([25.0, 25.0], 6.0), ([25.0, 5.0], 6.0),
                                           ([25.0, 15.0], 6.0), ([5.0, 25.0], 6.0),
                                           ([5.0, 15.0], 6.0), ([5.0, 5.0], 6.0)];
    let start = Node::new([0.0, 0.0], None);
    let end = Node::new([30.0, 30.0], None);
    let zs = zone_list.iter().map(|z| Zone::new(z.0, z.1)).collect_vec();
    let mut graph = Graph::build_graph(start, end, zs);
    let (came_from, _) = a_star(&mut graph);
    let f = File::create("json_out/graph.json").expect("Failed to create");
    serde_json::to_writer_pretty(f, &graph).expect("can't write");
    let found_path = reconstruct_path(came_from, start, end);
    let output_str = serde_json::to_string(&found_path);
    let f = File::create("json_out/found_path.json").expect("Failed to create");
    serde_json::to_writer_pretty(f, &found_path).expect("can't write");
    println!("{:?}", output_str);
}
