use std::{borrow::BorrowMut, collections::HashMap, fmt::Debug};

use itertools::Itertools;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use std::mem;
use uuid::Uuid;

#[derive(Debug)]
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

    pub fn neighbors(mut self, node: Node) -> Vec<Edge> {
        //There are three cases
        //First Case: node is start. We need to check to see if we had to escape
        //
        if node == self.start {
            println!("We are start");

            if self.edges.get(&node).unwrap().len() > 0 {
                println!("we have stuff in start")
            } else {
                let truth = line_of_sight_zones(&self.start, &self.end, &self.circles);
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

                    for tangent_pair in valid_tangents {
                        let edg = Edge::generate_edge(
                            self.start,
                            *tangent_pair.1,
                            Decimal::from_f32(f32::INFINITY).unwrap(),
                        );
                        self.edges
                            .entry(self.start)
                            .and_modify(|edges| edges.push(edg));
                    }
                } else {
                    println!("Generate Edges for end");

                    return vec![Edge::generate_edge(self.start, self.end, Decimal::MAX)];
                }
            }
        }
        return self.edges.get(&node).unwrap().to_vec();
    }
}
#[derive(Debug, Clone)]
pub struct Circle {
    location: Point,
    uuid: Uuid,
    radius: Decimal,
    nodes: Vec<Node>,
}

impl Circle {
    pub fn new(location: [f32; 2], radius: Decimal) -> Self {
        Self {
            location: Point::new(location[0], location[1]),
            uuid: Uuid::new_v4(),
            radius,
            nodes: Vec::<Node>::new(),
        }
    }
}

impl LocationRadius for Circle {
    fn loc_radius(&self) -> (Vec<Decimal>, Decimal) {
        (self.location.as_vec(), self.radius)
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Copy)]
pub struct Node {
    location: Point,
}
impl Node {
    pub fn new(location: [f32; 2]) -> Self {
        Self {
            location: Point::new(location[0], location[1]),
        }
    }
    pub fn from_dec(location: [Decimal; 2]) -> Self {
        Self {
            location: Point::from_dec(location[0], location[1]),
        }
    }
}

impl LocationRadius for Node {
    fn loc_radius(&self) -> (Vec<Decimal>, Decimal) {
        (self.location.as_vec(), dec!(0.0))
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, Copy)]
pub struct Point {
    x: Decimal,
    y: Decimal,
}

impl Point {
    fn new(x: f32, y: f32) -> Point {
        Point {
            x: Decimal::from_f32(x).expect("Unable to extract"),
            y: Decimal::from_f32(y).expect("Unable to extract"),
        }
    }

    fn from_dec(x: Decimal, y: Decimal) -> Self {
        Self { x, y }
    }

    fn as_vec(self) -> Vec<Decimal> {
        vec![self.x, self.y]
    }
}

trait LocationRadius {
    fn loc_radius(&self) -> (Vec<Decimal>, Decimal);
}

#[derive(Debug, Clone)]
pub struct Edge {
    node: Node,
    weight: Decimal,
    theta: Decimal,
    direction: Vec<Decimal>,
}

impl Edge {
    fn new(n: Node, w: Decimal, t: Decimal, d: Vec<Decimal>) -> Edge {
        Edge {
            node: n,
            weight: w,
            direction: d,
            theta: t,
        }
    }
    fn generate_edge(start: Node, end: Node, theta: Decimal) -> Edge {
        let start_loc = &start.location.as_vec();
        let end_loc = &end.location.as_vec();
        let distance = distance(&start_loc, &end_loc).expect("bad value");
        let comb_vec = subtrac_pts(&end_loc, &start_loc);
        let direction = comb_vec.iter().map(|val| val / distance).collect_vec();
        Edge::new(end, distance, theta, direction)
    }
}

//Pulled from old Rust std
// really want to use this but don't need it at this time
// fn integer_decode(val: f32) -> (u64, i16, i8) {
//     let bits: u64 = unsafe { mem::transmute(val as f64) };
//     let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
//     let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
//     let mantissa = if exponent == 0 {
//         (bits & 0xfffffffffffff) << 1
//     } else {
//         (bits & 0xfffffffffffff) | 0x10000000000000
//     };

//     exponent -= 1023 + 52;
//     (mantissa, exponent, sign)
// }

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
    let a = &node_1.location.as_vec();
    let b = &node_2.location.as_vec();
    let ab_difference = subtrac_pts(b, a);
    println!("ab_diff");
    dbg!(ab_difference.iter().map(|x| x.to_f64()));
    let ab_dot = dot_product(&ab_difference, &ab_difference);
    for zone in zones.iter() {
        let c = &zone.location.as_vec();
        let ac_difference = subtrac_pts(c, a);
        println!("ac_diff");
        dbg!(ac_difference.iter().map(|x| x.to_f64()));
        let u = dot_product(&ac_difference, &ab_difference) / ab_dot;

        // Clamp u and find e the point that intersects ab and passes through c
        dbg!(u.to_f64());
        dbg!(u.clamp(dec!(0.0), dec!(1.0)));
        let clamp_product: Vec<Decimal> = ab_difference
            .iter()
            .map(|value| value * u.clamp(dec!(0.0), dec!(1.0)))
            .collect();
        let e = add_pts(a, &clamp_product);
        let d = distance(c, &e).expect("Bad value");
        println!("distance and radius: {:?}, {:?}", d.to_f64(), zone.radius.to_f64());

        if d.to_f64() < zone.radius.to_f64() {
            return true;
        }
    }

    false
}

pub fn line_of_sight(node_1: &Node, node_2: &Node, zone: &Circle) -> bool {
    // Calculate u
    let a = &node_1.location.as_vec();
    let b = &node_2.location.as_vec();
    let ab_difference = subtrac_pts(b, a);
    let ab_dot = dot_product(&ab_difference, &ab_difference);
    let c = &zone.location.as_vec();
    let ac_difference = subtrac_pts(c, a);
    let u = dot_product(&ac_difference, &ab_difference) / ab_dot;

    // Clamp u and find e the point that intersects ab and passes through c
    let clamp_product: Vec<Decimal> = ab_difference
        .iter()
        .map(|value| value * u.clamp(dec!(0.0), dec!(1.0)))
        .collect();
    let e = add_pts(a, &clamp_product);
    let d = distance(c, &e).expect("distance failed");

    if d < zone.radius {
        return true;
    }

    false
}
pub fn dot_product(p1: &[Decimal], p2: &[Decimal]) -> Decimal {
    let dot: Decimal = p1.iter().zip(p2.iter()).map(|(a, b)| a * b).sum();
    dot
}

pub fn distance(p1: &[Decimal], p2: &[Decimal]) -> Option<Decimal> {
    println!("distance points");
    let p = dbg!(p1.iter().map(|x| x.to_f64().unwrap()).collect_vec());
    let y = dbg!(p2.iter().map(|x| x.to_f64().unwrap()).collect_vec());
    let square_sum: Decimal = p1
        .iter()
        .zip(p2.iter())
        .map(|(x1, x2)| (x2 - x1).powi(2))
        .sum();
    square_sum.sqrt()
}

pub fn add_pts(p1: &[Decimal], p2: &[Decimal]) -> Vec<Decimal> {
    println!("addpts");
    let p = dbg!(p1.iter().map(|x| x.to_f64().unwrap()).collect_vec());
    let y = dbg!(p2.iter().map(|x| x.to_f64().unwrap()).collect_vec());
    let point_sum: Vec<Decimal> = p1.iter().zip(p2.iter()).map(|(x1, x2)| x1 + x2).collect();
    point_sum
}

pub fn subtrac_pts(p1: &[Decimal], p2: &[Decimal]) -> Vec<Decimal> {
    println!("sub points");
    let p = dbg!(p1.iter().map(|x| x.to_f64().unwrap()).collect_vec());
    let y = dbg!(p2.iter().map(|x| x.to_f64().unwrap()).collect_vec());
    let difference: Vec<Decimal> = p1.iter().zip(p2.iter()).map(|(x1, x2)| x1 - x2).collect();
    difference
}
///https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Tangents_between_two_circles
fn generate_tangents(
    start_circle: (Vec<Decimal>, Decimal),
    end_circle: (Vec<Decimal>, Decimal),
) -> Vec<(Node, Node)> {
    let (start_loc, start_radius) = start_circle;
    let (end_loc, end_radius) = end_circle;
    let d = distance(&end_loc, &start_loc).expect("bad value in distance");
    let mut tangents: Vec<(Node, Node)> = Vec::new();
    //Here we want to do vector math for the tangents as a whole
    if d < start_radius - end_radius {
        return tangents;
    }
    let center_difference = subtrac_pts(&start_loc, &end_loc);
    let center_norm = center_difference.iter().map(|val| val / d).collect_vec();
    //TODO: Figure out what comparison to use for start/end

    for sign1 in (-1..1).step_by(2) {
        let dec_sign1 = Decimal::from_i32(sign1).unwrap();
        let c = (start_radius - dec_sign1 * end_radius) / d;
        if c.powi(2) > dec!(1.0) {
            continue;
        }
        let h = (dec!(1.0) - c * c)
            .sqrt()
            .expect("bad value")
            .max(dec!(0.0));
        for sign2 in (-1..1).step_by(2) {
            let dec_sign2 = Decimal::from_i32(sign2).unwrap();
            let nx = center_norm[0] * c - dec_sign2 * h * center_norm[1];
            let ny = center_norm[1] * c + dec_sign2 * h * center_norm[0];

            let tangent_1_loc = [
                start_loc[0] + start_radius * nx,
                start_loc[1] + start_radius * ny,
            ];
            let tangent_2_loc = [
                end_loc[0] + dec_sign1 * end_radius * nx,
                end_loc[1] + dec_sign1 * end_radius * ny,
            ];
            dbg!(tangent_1_loc);
            dbg!(tangent_2_loc);

            let tan_node_start = Node::from_dec(tangent_1_loc);
            let tan_node_end = Node::from_dec(tangent_2_loc);

            tangents.push((tan_node_start, tan_node_end));
        }
    }
    return tangents;
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_point() {
        let float_pt = dec!(2.333);
        let float_pt_2 = dec!(2.333);
        let pt = Point::from_dec(float_pt, float_pt_2);
        assert_eq!(pt.as_vec(), vec![float_pt, float_pt_2]);
        assert_eq!(pt.as_vec()[1], float_pt_2);
    }

    #[test]
    fn test_add_pt() {
        let sum = add_pts(&vec![dec!(0.0), dec!(2.0)], &vec![dec!(0.0), dec!(3.0)]);
        assert_eq!(sum, vec![dec!(0.0), dec!(5.0)])
    }
    #[test]
    fn test_distance() {
        let d = distance(&vec![dec!(0.0), dec!(0.0)], &vec![dec!(0.0), dec!(1.0)]).unwrap();
        assert_eq!(d, dec!(1.0))
    }

    #[test]
    fn test_line_of_sight() {
        let n1 = Node::new([0.0, 0.0]);
    }

    #[test]
    fn graph_build() {
        let start = Node::new([0.0, 0.0]);
        let end = Node::new([5.0, 5.0]);
        let circle = Circle::new([2.0, 2.0], dec!(2.0));
        let circle_vec = vec![circle];
        let graph = Graph::build_graph(start, end, circle_vec);
        let nodes = graph.neighbors(start);
        // assert_eq!(nodes.len(), 0);
        // print!("{:?}", graph)
    }

    #[test]
    fn simple_graph_no_circle() {
        let start = Node::new([0.0, 0.0]);
        let end = Node::new([5.0, 5.0]);
        let circle_vec = Vec::<Circle>::new();
        let graph = Graph::build_graph(start, end, circle_vec);
        let nodes = graph.neighbors(start);
        // assert_eq!(nodes.len(), 0);
        // print!("{:?}", graph)
    }
}
