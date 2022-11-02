use avoidrs::*;

#[test]
fn test_distance() {
    let d = distance(&vec![0.0, 0.0], &vec![0.0, 1.0]);
    assert_eq!(d, 1.0)
}

#[test]
fn test_line_of_sight() {}

#[test]
fn test_3d_distance() {
    let d = distance(&vec![0.0, 0.0, 0.0], &vec![1.0, 1.0, 1.0]);
    assert_eq!(d, 3_f64.sqrt())
}

#[test]
fn hypt_vs_distance() {
    let sum = subtrac_pts(&vec![0.0, 2.0], &vec![0.0, 3.0]);
    let d = distance(&vec![0.0, 0.0], &vec![0.0, 1.0]);
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
    let (came_from, cost) = a_star(&mut graph);
    let j = serde_json::to_string_pretty(&graph).expect("can't write");
    let path = reconstruct_path(came_from, start, end);
    // println!("path: {:?}", path);
    // println!("{}", j);
    // assert_eq!(nodes.unwrap().len(), 2);
    // print!("{:?}", graph)
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
    let (came_from, cost) = a_star(&mut graph);
    let path = reconstruct_path(came_from, start, end);
    assert_eq!(path.len(), 2)
    // print!("{:?}", graph)
}
