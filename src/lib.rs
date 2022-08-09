use std::{collections::HashMap, borrow::BorrowMut};

use itertools::Itertools;
use std::mem;

#[derive(Debug)]
pub struct Graph {
    start: Node,
    end: Node,
    circles: Vec<Circle>,
    // Edge: EndNode, weight, theta, direction
    edges: HashMap<Node, Vec<Edge>>
}

impl Graph {
    fn new(start: Node, end: Node, circles: Vec<Circle>) -> Self{
        Self{
            start,
            end,
            circles,
            edges: HashMap::from([(start, Vec::<Edge>::new())])
        }
    }
    fn neighbors(self, node: Node) -> Vec<Edge>{
        //There are three cases
        //First Case: node is start. We need to check to see if we had to escape
        //
        if node == self.start {
            println!("We are start");
            
            if self.edges.get(&node).unwrap().len() > 0 {
                println!("we have stuff in start")
            }else {
                println!("start is empty")
            }

        }
        return self.edges.get(&node).unwrap().to_vec();
    }
}
#[derive(Debug)]
pub struct Circle {
    location: [Point; 2],
    radius: f64,
    nodes: Vec<Node>,
}

impl Circle {
    fn new(location: [f64; 2], radius: f64) -> Self {
        Self {
            location: [Point::new(location[0]), Point::new(location[1])],
            radius,
            nodes: Vec::<Node>::new(),
        }
    }
}
#[derive(Debug, Clone, Hash, Eq, PartialEq, Copy)]
pub struct Node {
    location: [Point; 2],
}
impl Node {
    fn new(location: [f64; 2]) -> Self {
        Self {
            location: [Point::new(location[0]), Point::new(location[1])] 
        }
    }
}

#[derive(Clone,Hash, Debug, PartialEq, Eq, Copy)]
pub struct Point((u64, i16, i8));

impl Point{
    fn new(val: f64) -> Point {
        Point(integer_decode(val))
    }
    fn float_encode(self) -> f64 {
        (self.0.0 as f64) * (self.0.1 as f64).exp2() * self.0.2 as f64
    }
}

#[derive(Debug, Clone)]
pub struct Edge {
    edge_type: EdgeType,
    weight: f32,
    theta: f32,
    direction: Vec<f32>,
}

impl Edge {
    fn new(edge_type: EdgeType, weight: f32, theta: f32, direction: Vec<f32>) -> Self {
        Self { edge_type, weight, theta, direction }
    }
}
#[derive(Debug, Clone)]
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
) -> Graph {
    Graph::new(start, end, zones)
}

//Pulled from old Rust std
fn integer_decode(val: f64) -> (u64, i16, i8) {
    let bits: u64 = unsafe { mem::transmute(val) };
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
/// let result = line_of_sight(node_1, node_2, zone);
/// assert_eq!(result, );
/// ```
pub fn line_of_sight_zones(node_1: &Node, node_2: &Node, zones: Vec<Circle>) -> bool {
    // Calculate u
    let a = &node_1.location.iter().map(|value| value.float_encode()).collect_vec();
    let b = &node_2.location.iter().map(|value| value.float_encode()).collect_vec();
    let ab_difference = subtrac_pts(b, a);
    let ab_dot = dot_product(&ab_difference, &ab_difference);
    for zone in zones {
        let c = &zone.location.iter().map(|value| value.float_encode()).collect_vec();
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
    let a = &node_1.location.iter().map(|value| value.float_encode()).collect_vec();
    let b = &node_2.location.iter().map(|value| value.float_encode()).collect_vec();
    let ab_difference = subtrac_pts(b, a);
    let ab_dot = dot_product(&ab_difference, &ab_difference);
    let c = &zone.location.iter().map(|value| value.float_encode()).collect_vec();
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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_point() {
        let float_pt = 2.333;
        let pt = Point::new(float_pt);
        assert_eq!(pt.float_encode(), float_pt)
    }

    #[test]
    fn graph_build() {
        let start = Node::new([0.0, 0.0]);
        let end = Node::new([5.0, 5.0]);
        let circle = Circle::new([2.0, 2.0], 2.0);
        let circle_vec = vec![circle];
        let graph = build_graph(start, end, circle_vec);
        let nodes = graph.neighbors(start);
        assert_eq!(nodes.len(), 0);
        // print!("{:?}", graph)
    }

    #[test]
    fn simple_graph_no_circle() {
        let start = Node::new([0.0, 0.0]);
        let end = Node::new([5.0, 5.0]);
        // print!("{:?}", graph)
    }
}
