use avoidrs::{reconstruct_path, Graph, Node, Zone};

fn main() {
    let start = Node::new([0.0, 0.0], None);
    let end = Node::new([30.0, 30.0], None);
    let circle_vec = Vec::<Zone>::new();
    let graph = Box::new(Graph::build_graph(start, end, circle_vec));
    let (came_from, cost) = graph.a_star();
    let path = reconstruct_path(came_from, start, end);
}
