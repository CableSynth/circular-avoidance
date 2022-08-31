use std::{borrow::BorrowMut, collections::HashMap};
use priority_queue::PriorityQueue;
use itertools::Itertools;
use std::mem;
use uuid::Uuid;
use core::cmp::Ordering;

#[derive(Debug, Clone, )]
pub struct Graph {
    start: Node,
    end: Node,
    circles: Vec<Circle>,
    // Edge: EndNode, weight, theta, direction
    edges: HashMap<Node, Vec<Edge>>,
}

impl Graph {
    pub fn build_graph(start: Node, end: Node, zones: Vec<Circle>) -> Graph {
        Graph::new(start, end, zones)
    }
    fn new(start: Node, end: Node, circles: Vec<Circle>) -> Self {
        Self {
            start,
            end,
            circles,
            edges: HashMap::from([(start, Vec::<Edge>::new())]),
        }
    }

    pub fn neighbors(mut self, node: Node) -> Option<Vec<Edge>> {
        //There are three cases
        //First Case: node is start. We need to check to see if we had to escape
        //Second Case: we are at a zone. Find all bitangents from zone to others
        //3rd Case: traversal of nodes avaliable

        //Always check for items at the node
        if self.edges.get(&node).unwrap().len() > 0 {
            println!("we have items in the node");
            return Some(self.edges.get(&node).unwrap().to_vec());
        } else if node == self.start {
            println!("We are start");

            //Check if we escaped
            let truth = line_of_sight_zones(&self.start, &self.end, &self.circles);

            //are any circles in our way
            if truth {
                println!("Build bitangents for all zones from start");
                let possible_tangents = self
                    .circles
                    .iter()
                    .flat_map(|c| generate_tangents(self.start.loc_radius(), c.loc_radius()))
                    .collect_vec();
                let temp_c = Vec::from(self.circles);
                let valid_tangents = possible_tangents
                    .iter()
                    .filter_map(|(s, e)| {
                        if line_of_sight_zones(s, e, &temp_c) {
                            None
                        } else {
                            Some((s, e))
                        }
                    })
                    .collect_vec();
                //Input all tangent pairs into the graph from start
                //Build and add new empty entry to access later
                for tangent_pair in valid_tangents {
                    let edg = Edge::generate_edge(self.start, *tangent_pair.1, f64::INFINITY);
                    self.edges
                        .entry(self.start)
                        .and_modify(|edges| edges.push(edg));
                    self.edges.insert(*tangent_pair.1, Vec::<Edge>::new());
                }
                dbg!(&self.edges);
            } else {
                // we can go directly to end
                println!("Generate Edges for end");

                return Some(vec![Edge::generate_edge(self.start, self.end, f64::INFINITY)]);
            }
        } else {
            // need to get the circle that node lies on
        }
        return Some(self.edges.get(&node).unwrap().to_vec());
    }

    pub fn a_star(&self) {
        let mut frontier: PriorityQueue<Node, Number> = PriorityQueue::new();
        frontier.push(self.start, Number(0.0));
        let mut came_from: HashMap<Node, Node> = HashMap::new();
        let mut cost_so_far: HashMap<Node, f32> = HashMap::from([(self.start, 0.0)]);

        while !frontier.is_empty() {
            let (current, _ )= frontier.pop().expect("No poped off an empty q");

            // if current == self.end {
            //     println!("we have reached the end!");
            //     return ;
            // }

            for edge in self.neighbors(current) {

            }

        }

    }
}
#[derive(Debug, Clone)]
pub struct Circle {
    location: Point,
    uuid: Uuid,
    radius: f64,
    nodes: Vec<Node>,
}

impl Circle {
    pub fn new(location: [f64; 2], radius: f64) -> Self {
        Self {
            location: Point::new(location[0], location[1]),
            uuid: Uuid::new_v4(),
            radius,
            nodes: Vec::<Node>::new(),
        }
    }
}

impl LocationRadius for Circle {
    fn loc_radius(&self) -> (Vec<f64>, f64, Option<Uuid>) {
        (self.location.float_encode(), self.radius, Some(self.uuid))
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Copy)]
pub struct Node {
    location: Point,
    circle: Uuid,
}
impl Node {
    pub fn new(location: [f64; 2], id: Option<Uuid>) -> Self {
        Self {
            location: Point::new(location[0], location[1]),
            circle: id.unwrap_or_default(),
        }
    }
}

impl LocationRadius for Node {
    fn loc_radius(&self) -> (Vec<f64>, f64, Option<Uuid>) {
        (self.location.float_encode(), 0.0, None)
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, Copy)]
pub struct Point {
    x: (u64, i16, i8),
    y: (u64, i16, i8),
}

impl Point {
    fn new(x: f64, y: f64) -> Point {
        Point {
            x: integer_decode(x),
            y: integer_decode(y),
        }
    }
    fn float_encode(self) -> Vec<f64> {
        vec![
            ((self.x.0 as f64) * (self.x.1 as f64).exp2() * self.x.2 as f64),
            ((self.y.0 as f64) * (self.y.1 as f64).exp2() * self.y.2 as f64),
        ]
    }
}

trait LocationRadius {
    fn loc_radius(&self) -> (Vec<f64>, f64, Option<Uuid>);
}

#[derive(Debug, Clone)]
pub struct Edge {
    node: Node,
    weight: f64,
    theta: f64,
    direction: Vec<f64>,
}

impl Edge {
    fn new(node: Node, weight: f64, theta: f64, direction: Vec<f64>) -> Self {
        Self {
            node,
            weight,
            theta,
            direction,
        }
    }
    fn generate_edge(start: Node, end: Node, theta: f64) -> Edge {
        let start_loc = &start.location.float_encode();
        let end_loc = &end.location.float_encode();
        let distance = distance(&start_loc, &end_loc);
        let comb_vec = subtrac_pts(&end_loc, &start_loc);
        let direction = comb_vec.iter().map(|val| val / distance).collect_vec();
        Edge::new(end, distance, theta, direction)
    }
}

#[derive(PartialEq, PartialOrd)]
pub struct Number(f64);

impl Eq for Number {}

impl Ord for Number{
    fn cmp(&self, other: &Self) -> Ordering {
	if let Some(ordering) = self.partial_cmp(other) {
	    ordering
	} else {
	    // Choose what to do with NaNs, for example:
	    Ordering::Less
	}
    }
}
//Pulled from old Rust std
fn integer_decode(val: f64) -> (u64, i16, i8) {
    let bits: u64 = unsafe { mem::transmute(val as f64) };
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };

    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}

///
/// # Arguments
/// * `node_1`
/// * `node_2`
/// * `zone`
/// This functions takes in two points and a zone
/// # Examples
///
/// ```
/// use circular_avoidance::line_of_sight;
///
/// let result = line_of_sight(node_1, node_2, zones);
/// assert_eq!(result, );
/// ```
pub fn line_of_sight_zones(node_1: &Node, node_2: &Node, zones: &[Circle]) -> bool {
    // Calculate u
    let a = &node_1.location.float_encode();
    let b = &node_2.location.float_encode();
    let ab_difference = subtrac_pts(b, a);
    let ab_dot = dot_product(&ab_difference, &ab_difference);
    for zone in zones.iter() {
        let c = &zone.location.float_encode();
        let ac_difference = subtrac_pts(c, a);
        let u = dot_product(&ac_difference, &ab_difference) / ab_dot;

        // Clamp u and find e the point that intersects ab and passes through c
        let clamp_product: Vec<f64> = ab_difference
            .iter()
            .map(|value| value * u.clamp(0.0, 1.0))
            .collect();
        let e = add_pts(a, &clamp_product);
        let d = distance(c, &e);

        if d < zone.radius {
            return true;
        }
    }

    false
}

pub fn line_of_sight(node_1: &Node, node_2: &Node, zone: &Circle) -> bool {
    // Calculate u
    let a = &node_1.location.float_encode();
    let b = &node_2.location.float_encode();
    let ab_difference = subtrac_pts(b, a);
    let ab_dot = dot_product(&ab_difference, &ab_difference);
    let c = &zone.location.float_encode();
    let ac_difference = subtrac_pts(c, a);
    let u = dot_product(&ac_difference, &ab_difference) / ab_dot;

    // Clamp u and find e the point that intersects ab and passes through c
    let clamp_product: Vec<f64> = ab_difference
        .iter()
        .map(|value| value * u.clamp(0.0, 1.0))
        .collect();
    let e = add_pts(a, &clamp_product);
    let d = distance(c, &e);

    if d < zone.radius {
        return true;
    }

    false
}
pub fn dot_product(p1: &[f64], p2: &[f64]) -> f64 {
    let dot: f64 = p1.iter().zip(p2.iter()).map(|(a, b)| a * b).sum();
    dot
}

pub fn distance(p1: &[f64], p2: &[f64]) -> f64 {
    let square_sum: f64 = p1
        .iter()
        .zip(p2.iter())
        .map(|(x1, x2)| (x2 - x1).powi(2))
        .sum();
    square_sum.sqrt()
}

pub fn add_pts(p1: &[f64], p2: &[f64]) -> Vec<f64> {
    let point_sum: Vec<f64> = p1.iter().zip(p2.iter()).map(|(x1, x2)| x1 + x2).collect();
    point_sum
}

pub fn subtrac_pts(p1: &[f64], p2: &[f64]) -> Vec<f64> {
    let difference: Vec<f64> = p1.iter().zip(p2.iter()).map(|(x1, x2)| x1 - x2).collect();
    difference
}
///https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Tangents_between_two_circles
/// TODO: use Option as return
fn generate_tangents(
    start_circle: (Vec<f64>, f64, Option<Uuid>),
    end_circle: (Vec<f64>, f64, Option<Uuid>),
) -> Vec<(Node, Node)> {
    let (start_loc, start_radius, start_uuid) = start_circle;
    let (end_loc, end_radius, end_uuid) = end_circle;
    let d = distance(&end_loc, &start_loc);
    let mut tangents: Vec<(Node, Node)> = Vec::new();
    //Here we want to do vector math for the tangents as a whole
    if d < start_radius - end_radius {
        return tangents;
    }
    let center_difference = subtrac_pts(&start_loc, &end_loc);
    let center_norm = center_difference.iter().map(|val| val / d).collect_vec();
    //TODO: Figure out what comparison to use for start/end

    for sign1 in (-1..2).step_by(2) {
        let mut c = (start_radius - sign1 as f64 * end_radius) / d;
        c = round_to(c);
        if c.powi(2) > 1.0 {
            continue;
        }
        let mut h = (1.0 - c * c).sqrt().max(0.0);
        h = round_to(h);
        for sign2 in (-1..2).step_by(2) {
            let nx = center_norm[0] * c - sign2 as f64 * h as f64 * center_norm[1];
            let ny = center_norm[1] * c + sign2 as f64 * h as f64 * center_norm[0];

            let tangent_1_loc = [
                round_to(start_loc[0] + start_radius * nx),
                round_to(start_loc[1] + start_radius * ny),
            ];
            let tangent_2_loc = [
                round_to(end_loc[0] - sign1 as f64 * end_radius * nx),
                round_to(end_loc[1] - sign1 as f64 * end_radius * ny),
            ];

            let tan_node_start = Node::new(tangent_1_loc, start_uuid);
            let tan_node_end = Node::new(tangent_2_loc, end_uuid);

            tangents.push((tan_node_start, tan_node_end));
        }
        if start_radius == 0.0 {
            return tangents;
        }
    }
    return tangents;
}

fn round_to(num: f64) -> f64 {
    return (num * 10000.0).round() / 10000.0;
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_point() {
        let float_pt = 2.333;
        let float_pt_2 = 2.333;
        let pt = Point::new(float_pt, float_pt_2);
        assert_eq!(pt.float_encode(), vec![float_pt, float_pt_2]);
        assert_eq!(pt.float_encode()[1], float_pt_2);
    }

    #[test]
    fn test_add_pt() {
        let sum = add_pts(&vec![0.0, 2.0], &vec![0.0, 3.0]);
        assert_eq!(sum, vec![0.0, 5.0])
    }
    #[test]
    fn test_distance() {
        let d = distance(&vec![0.0, 0.0], &vec![0.0, 1.0]);
        assert_eq!(d, 1.0)
    }

    #[test]
    fn test_line_of_sight() {
        let n1 = Node::new([0.0, 0.0], None);
    }

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
        let end = Node::new([5.0, 5.0], None);
        let circle = Circle::new([2.0, 2.0], 2.0);
        let circle_vec = vec![circle];
        let graph = Graph::build_graph(start, end, circle_vec);
        let nodes = graph.neighbors(start);
        assert_eq!(nodes.len(), 2);
        // print!("{:?}", graph)
    }

    #[test]
    fn simple_graph_no_circle() {
        let start = Node::new([0.0, 0.0], None);
        let end = Node::new([5.0, 5.0], None);
        let circle_vec = Vec::<Circle>::new();
        let graph = Graph::build_graph(start, end, circle_vec);
        let nodes = graph.neighbors(start);
        assert_eq!(nodes.len(), 1);
        // print!("{:?}", graph)
    }
}
