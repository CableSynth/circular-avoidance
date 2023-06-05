use avoidrs::*;
use itertools::Itertools;
use std::fs::File;

#[test]
fn test_distance() {
    let d = distance(&[0.0, 0.0], &[0.0, 1.0]);
    assert_eq!(d, 1.0)
}

#[test]
fn test_line_of_sight() {
    let start = Node::new([0.0, 0.0], None);
    let end = Node::new([6.5, 7.4], None);
    let z = Zone::new([3.0, 4.0], 2.0);
    let los = line_of_sight(&start, &end, &z);
    assert!(los)
}

#[test]
fn test_3d_distance() {
    let d = distance(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
    assert_eq!(d, 3_f64.sqrt())
}

#[test]
fn hypt_vs_distance() {
    let sum = subtrac_pts(&[0.0, 2.0], &[0.0, 3.0]);
    let d = distance(&[0.0, 0.0], &[0.0, 1.0]);
    let h = sum[0].hypot(sum[1]);
    assert_eq!(d, h)
}
#[test]
fn graph_build() {
    let start = Node::new([0.0, 0.0], None);
    let end = Node::new([30.0, 30.0], None);
    let z = Zone::new([2.0, 2.0], 2.0);
    let z1 = Zone::new([9.0, 9.0], 3.0);
    let z2 = Zone::new([15.0, 15.0], 6.0);
    let circle = vec![z, z1, z2];
    let mut graph = Graph::build_graph(start, end, circle);
    let (came_from, _cost) = a_star(&mut graph);
    let _path = reconstruct_path(came_from, start, end);
    println!("path: {:?}", _path);
    // println!("{}", j);
    // assert_eq!(nodes.unwrap().len(), 2);
    // print!("{:?}", graph)
}

#[test]
fn square_zones() {
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
    let f = File::create("json_out/found_path.json").expect("Failed to create");
    serde_json::to_writer_pretty(f, &found_path).expect("can't write");
}

#[test]
fn simple_graph_no_circle() {
    let start = Node::new([0.0, 0.0], None);
    let end = Node::new([5.0, 5.0], None);
    let circle_vec = Vec::<Zone>::new();
    let mut graph = Graph::build_graph(start, end, circle_vec);
    let nodes = graph.neighbors(start);
    assert_eq!(nodes.unwrap().len(), 1);
    // print!("{:?}", graph)
}

#[test]
fn simple_graph_no_circle_path() {
    let start = Node::new([0.0, 0.0], None);
    let end = Node::new([5.0, 5.0], None);
    let circle_vec = Vec::<Zone>::new();
    let mut graph = Graph::build_graph(start, end, circle_vec);
    let (came_from, _cost) = a_star(&mut graph);
    let path = reconstruct_path(came_from, start, end);
    assert_eq!(path.len(), 2)
    // print!("{:?}", graph)
}
