use core::cmp::Ordering;
use core::fmt;
use itertools::Itertools;
use priority_queue::PriorityQueue;
use serde::__private::de;
use serde::ser::SerializeStruct;
use serde::{ser, Serialize};
use serde_json_any_key::*;
use std::collections::HashMap;
use std::ops::Add;
use std::{f64::consts::PI, vec};
use uuid::Uuid;
use robust::Coord;

#[derive(Debug, Clone, Serialize)]
pub struct Graph {
    start: Node,
    end: Node,
    circles: HashMap<Uuid, Zone>,
    /// Edge: EndNode, weight, theta, direction
    #[serde(with = "any_key_map")]
    edges: HashMap<Node, Vec<Edge>>,
}

impl Graph {
    pub fn build_graph(start: Node, end: Node, zones: Vec<Zone>) -> Graph {
        let zones_map: HashMap<Uuid, Zone> = zones.iter().cloned().map(|c| (c.uuid, c)).collect();
        Graph::new(start, end, zones_map)
    }
    fn new(start: Node, end: Node, circles: HashMap<Uuid, Zone>) -> Self {
        Self {
            start,
            end,
            circles,
            edges: HashMap::from([(start, Vec::<Edge>::new()), (end, Vec::<Edge>::new())]),
        }
    }

    pub fn neighbors(&mut self, node: Node) -> Option<Vec<Edge>> {
        //There are three cases
        //First Case: node is start. We need to check to see if we had to escape
        //Second Case: we are at a zone. Find all bitangents from zone to others
        //3rd Case: traversal of nodes avaliable

        //Always check for items at the node
        if self.edges.get(&node).is_none() {
            return None;
        } else if !self.edges.get(&node).expect("No node in graph").is_empty() {
            return Some(self.edges.get(&node)?.to_vec());
        } else if node == self.start {
            let is_blocked = line_of_sight_zones(
                &self.start,
                &self.end,
                &self.circles.values().cloned().collect_vec(),
            );

            //are any circles in our way
            if is_blocked {
                let temp_c = self.circles.values().cloned().collect_vec();

                let valid_tangents = tangent_prep(temp_c, self.start.loc_radius());
                //Input all tangent pairs into the graph from start
                //Build and add new empty entry to access later
                for tangent_pair in valid_tangents {
                    let edg = Edge::generate_edge(self.start, tangent_pair.1, f64::INFINITY, None);
                    self.edges
                        .entry(self.start)
                        .and_modify(|edges| edges.push(edg));
                    self.edges.insert(tangent_pair.1, Vec::<Edge>::new());
                }
            } else {
                // we can go directly to end

                return Some(vec![Edge::generate_edge(
                    self.start,
                    self.end,
                    f64::INFINITY,
                    None,
                )]);
            }
        } else {
            // need to get the circle that node lies on
            let circle_id = node.circle;
            let circle_of_node = self
                .circles
                .get_mut(&circle_id)
                .expect("Zone not valid?")
                .clone();
            //Find the valid circles (i.e all the ones except the one we are on)
            let valid_circles: Vec<Zone> = self
                .circles
                .iter()
                .filter_map(|x| {
                    if *x.0 != circle_id {
                        Some(x.1.clone())
                    } else {
                        None
                    }
                })
                .collect();

            let end_tangents =
                generate_tangents(self.end.loc_radius(), circle_of_node.loc_radius());
            let end_tangents = end_tangents
                .iter()
                .filter_map(|(start, end)| {
                    if line_of_sight_zones(&self.end, end, &valid_circles) {
                        None
                    } else {
                        Some((start, end))
                    }
                })
                .collect_vec();

            if end_tangents.is_empty() {
                let valid_tangents =
                    tangent_prep(valid_circles.clone(), circle_of_node.loc_radius());

                // We build out the surfing edges from the circles
                for tangent_pair in valid_tangents {
                    let edg =
                        Edge::generate_edge(tangent_pair.0, tangent_pair.1, f64::INFINITY, None);
                    println!("adding items to edge {:?}", (tangent_pair.0.location.float_encode(), tangent_pair.1.location.float_encode()));
                    self.circles
                        .entry(circle_id)
                        .and_modify(|z| z.nodes.push(tangent_pair.0));
                    let vec_for_edge: Vec<Edge> = vec![edg];
                    self.edges.insert(tangent_pair.0, vec_for_edge);
                    self.edges.insert(tangent_pair.1, Vec::<Edge>::new());
                    for edg in self.edges.get(&tangent_pair.0).unwrap() {
                        println!("\tedge {:?}", edg.node.location.float_encode());
                    }
                }
            } else {
                for (_, &n) in end_tangents {
                    let edg = Edge::generate_edge(n, self.end, f64::INFINITY, None);
                    println!("adding items to edge from end {:?}", (node.location.float_encode()));
                    self.circles
                        .entry(circle_id)
                        .and_modify(|z| z.nodes.push(n));
                    let vec_for_edge: Vec<Edge> = vec![edg];
                    self.edges.insert(n, vec_for_edge);
                }
            }
            //Next Build Huggin edges
            self.generate_hugging(node, &circle_of_node, valid_circles.clone());
        }
        return Some(self.edges.get(&node).unwrap().clone().to_vec());
    }

    fn generate_hugging(&mut self, node: Node, zone: &Zone, zone_vec: Vec<Zone>) {
        let radius_sqr = (zone.radius.powi(2)) * 2.0;
        let tangent_nodes = zone.nodes.clone();
        let intersects = self.hugging_edge_zone_intersection(zone_vec, zone);
        // Here is where we-wa-chu find the list of intersection areas
        // then we can filter the nodes in `tangent_nodes`
        let valid_nodes = self.cull_hugging(node, &tangent_nodes, intersects, zone.location);

        for n in valid_nodes {
            let dist = distance(&node.location.float_encode(), &n.location.float_encode());
            let round_temp = round_to((radius_sqr - dist.powi(2)) / radius_sqr, 5);
            let alpha = round_temp.acos();
            let edg = Edge::generate_edge(node, n, alpha, Some(zone.radius));
            self.edges.entry(node).and_modify(|edges| edges.push(edg));
        }
    }
    fn hugging_edge_zone_intersection(
        &mut self,
        zones: Vec<Zone>,
        focus: &Zone,
    ) -> Vec<(f64, f64)> {
        let (focus_loc, focus_radius, _) = focus.loc_radius();
        let intersections = zones
            .iter()
            .cloned()
            .filter_map(|z| {
                let (target_loc, target_radius, _) = z.loc_radius();
                let dist = distance(&target_loc, &focus_loc);
                let radius_sum = target_radius + focus_radius;
                let abs_radius_diff = (target_radius - focus_radius).abs();
                if dist > radius_sum
                    || dist < abs_radius_diff
                    || dist == radius_sum
                    || ((dist == 0.0) && (abs_radius_diff == 0.0))
                {
                    None
                } else {
                    let distance_a_c = (focus_radius.powi(2) - target_radius.powi(2)
                        + dist.powi(2))
                        / (2.0 * dist);
                    let h = focus_radius.powi(2) - distance_a_c.powi(2);
                    let temp_diff = subtrac_pts(&target_loc, &focus_loc);
                    let distance_div = distance_a_c / dist;
                    let center_mult = temp_diff.iter().map(|i| i * distance_div).collect_vec();
                    let intermediate = add_pts(&center_mult, &focus_loc);

                    let point_a_x = intermediate[0] + h * temp_diff[1] / dist;
                    let point_a_y = intermediate[1] - h * temp_diff[0] / dist;
                    let point_b_x = intermediate[0] - h * temp_diff[1] / dist;
                    let point_b_y = intermediate[1] + h * temp_diff[0] / dist;
                    // Put it here lol do the calculation for intersections angles

                    let intersection_edge_1 = subtrac_pts(&[point_a_x, point_a_y], &focus_loc);
                    let intersection_edge_2 = subtrac_pts(&[point_b_x, point_b_y], &focus_loc);
                    let intersection_angle_1 =
                        (-intersection_edge_1[1]).atan2(-intersection_edge_1[0]) + PI;
                    let intersection_angle_2 =
                        (-intersection_edge_2[1]).atan2(-intersection_edge_2[0]) + PI;
                    Some((intersection_angle_1, intersection_angle_2))
                }
            })
            .collect_vec();
        intersections
    }

    //Return the valide bitangents that create a hugging edge
    fn cull_hugging(
        &mut self,
        node: Node,
        nodes: &[Node],
        intersects: Vec<(f64, f64)>,
        zone_loc: Point,
    ) -> Vec<Node> {
        let arc_side1 = subtrac_pts(&node.location.float_encode(), &zone_loc.float_encode());
        let angle_1 = (-arc_side1[1]).atan2(-arc_side1[0]) + PI;
        let valid = nodes
            .iter()
            .filter_map(|n| {
                let arc_side2 = subtrac_pts(&n.location.float_encode(), &zone_loc.float_encode());
                let angle_2 = (-arc_side2[1]).atan2(-arc_side2[0]) + PI;
                let mut sorted_angles = [angle_1, angle_2];
                sorted_angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
                // Put the intesection Cull here
                self.intersection_cull(n, sorted_angles, &intersects)
            })
            .collect_vec();

        valid
    }

    fn intersection_cull(
        &mut self,
        node: &Node,
        sorted_angles: [f64; 2],
        intersections: &Vec<(f64, f64)>,
    ) -> Option<Node> {
        for inter in intersections {
            if (sorted_angles[0] < inter.0 && inter.0 < sorted_angles[1])
                || (sorted_angles[0] < inter.1 && inter.1 < sorted_angles[1])
            {
                return None;
            }
        }
        Some(*node)
    }
}
pub fn reconstruct_path(
    came_from: HashMap<Node, (Node, f64)>,
    start: Node,
    end: Node,
) -> Vec<(Node, f64)> {
    let mut current = (end, f64::INFINITY);
    let mut path: Vec<(Node, f64)> = Vec::new();
    while !current.0.eq(&start) {
        path.push(current);
        if let Some(c) = came_from.get(&current.0) {
            current = *c;
        } else {
            println!("No path found");
            break;
        };
    }
    path.push((start, f64::INFINITY));
    path.reverse();
    path
}

#[derive(Debug, Clone, Serialize)]
pub struct Zone {
    location: Coord<f64>,
    uuid: Uuid,
    radius: f64,
    //Thinking of reworking this to store in a different way
    nodes: Vec<Node>,
}

impl Zone {
    pub fn new(location: [f64; 2], radius: f64) -> Self {
        Self {
            location: Coord { x: location[0], y: location[1] },
            uuid: Uuid::new_v4(),
            radius,
            nodes: Vec::<Node>::new(),
        }
    }
}

impl LocationRadius for Zone {
    fn loc_radius(&self) -> (Coord<f64>, f64, Option<Uuid>) {
        (self.location, self.radius, Some(self.uuid))
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Copy, Serialize)]
pub struct Node {
    location: Coord<f64>,
    circle: Uuid,
}
impl Node {
    pub fn new(location: [f64; 2], id: Option<Uuid>) -> Self {
        Self {
            location: Coord { x: location[0], y: location[1] },
            circle: id.unwrap_or_default(),
        }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let loc_print = self.location;
        write!(f, "location: {:?}, circle {}", loc_print, self.circle)
    }
}

impl LocationRadius for Node {
    fn loc_radius(&self) -> (Vec<f64>, f64, Option<Uuid>) {
        (self.location, 0.0, None)
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
    fn encode_as_tuple(self) -> (f32, f32) {
        (
            ((self.x.0 as f64) * (self.x.1 as f64).exp2() * self.x.2 as f64) as f32,
            ((self.y.0 as f64) * (self.y.1 as f64).exp2() * self.y.2 as f64) as f32,
        )
    }
    
}

impl Add for Point {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let s = self.float_encode();
        let r = rhs.float_encode();
        let point_vec = s
            .iter()
            .zip(r.iter())
            .map(|(left, right)| left + right)
            .collect_vec();
        Point::new(point_vec[0], point_vec[1])
    }
}

#[derive(Serialize)]
#[serde(remote = "Coord<f64>")]
struct CoordDef {
    x: f64,
    y: f64,
}

trait LocationRadius {
    fn loc_radius(&self) -> (Vec<f64>, f64, Option<Uuid>);
}

#[derive(Debug, Clone, Serialize)]
pub struct Edge {
    node: Node,
    weight: f64,
    theta: f64,
    direction: Option<Vec<f64>>,
}

impl Edge {
    fn new(node: Node, weight: f64, theta: f64, direction: Option<Vec<f64>>) -> Self {
        Self {
            node,
            weight,
            theta,
            direction,
        }
    }
    fn generate_edge(start: Node, end: Node, theta: f64, radius: Option<f64>) -> Edge {
        let start_loc = &start.location;
        let end_loc = &end.location;
        let dist: f64;
        let comb_vec: Vec<f64>;
        let direction: Option<Vec<f64>>;
        if theta.is_infinite() {
            dist = distance(start_loc, end_loc);
            comb_vec = subtrac_pts(end_loc, start_loc);
            direction = Some(comb_vec.iter().map(|val| val / dist).collect_vec());
        } else {
            dist = radius.unwrap() * theta;
            direction = None;
        }
        Edge::new(end, dist, theta, direction)
    }
}

//Pulled from old Rust std
fn integer_decode(val: f64) -> (u64, i16, i8) {
    let bits: u64 = val.to_bits();
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

pub fn a_star(graph: &mut Graph) -> (HashMap<Node, (Node, f64)>, HashMap<Node, f64>) {
    let mut frontier: PriorityQueue<Node, Number> = PriorityQueue::new();
    frontier.push(graph.start, Number(0.0));
    let mut came_from: HashMap<Node, (Node, f64)> = HashMap::new();
    let mut cost_so_far: HashMap<Node, f64> = HashMap::from([(graph.start, 0.0)]);

    while !frontier.is_empty() {
        let (current, _) = frontier.pop().expect("an empty q");

        if current == graph.end {
            println!("we have reached the end!");
            break;
        }

        for e in graph.neighbors(current).unwrap_or_default() {
            let new_cost = cost_so_far.get(&current).expect("node not in cost") + e.weight;

            if !cost_so_far.contains_key(&e.node) || &new_cost < cost_so_far.get(&e.node).unwrap() {
                cost_so_far.insert(e.node, new_cost);
                let prio = new_cost
                    + distance(
                        &e.node.location.float_encode(),
                        &graph.clone().end.location.float_encode(),
                    );
                frontier.push(e.node, Number(prio));
                came_from.insert(e.node, (current, e.theta));
            }
        }
    }

    (came_from, cost_so_far)
}

pub fn line_of_sight_zones(node_1: &Node, node_2: &Node, zones: &[Zone]) -> bool {
    // calculate u
    let a = &node_1.location.float_encode();
    let b = &node_2.location.float_encode();
    let ab_difference = subtrac_pts(b, a);
    let ab_dot = dot_product(&ab_difference, &ab_difference);
    for zone in zones.iter() {
        let c = &zone.location.float_encode();
        let ac_difference = subtrac_pts(c, a);
        let u = round_to(dot_product(&ac_difference, &ab_difference) / ab_dot, 5);

        // Clamp u and find e the point that intersects ab and passes through c
        let clamp_product: Vec<f64> = ab_difference
            .iter()
            .map(|value| value * u.clamp(0.0, 1.0))
            .collect();
        let e = add_pts(a, &clamp_product);
        let d = round_to(distance(c, &e), 4);

        if d < round_to(zone.radius, 5) {
            return true;
        }
    }

    false
}

pub fn line_of_sight(node_1: &Node, node_2: &Node, zone: &Zone) -> bool {
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

    if round_to(d, 5) < round_to(zone.radius, 5) {
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

//TODO CHANGE NAME
//Move to Zone struct?
fn tangent_prep(
    zones: Vec<Zone>,
    loc_radius_oi: (Vec<f64>, f64, Option<Uuid>),
) -> Vec<(Node, Node)> {
    let possible_tangents = zones
        .iter()
        .flat_map(|c| generate_tangents(loc_radius_oi.clone(), c.loc_radius()))
        .collect_vec();

    println!("possible tangents: {:?}", possible_tangents.iter().map(|(s,e)| (s.location.float_encode(), e.location.float_encode())).collect_vec());
    let valid_tangents = possible_tangents
        .iter()
        .filter_map(|(s, e)| {
            if line_of_sight_zones(s, e, &zones) {
                None
            } else {
                Some((*s, *e))
            }
        })
        .collect_vec();
    println!("valid tangents: {:?}", valid_tangents.iter().map(|(s,e)| (s.location.float_encode(), e.location.float_encode())).collect_vec());
    valid_tangents
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

    for sign1 in (-1..=1).step_by(2) {
        let mut c = (start_radius - sign1 as f64 * end_radius) / d;
        c = round_to(c, 5);
        if c.powi(2) > 1.0 {
            continue;
        }
        let mut h = (1.0 - c * c).sqrt().max(0.0);
        h = round_to(h, 5);
        for sign2 in (-1..2).step_by(2) {
            let nx = center_norm[0] * c - sign2 as f64 * h * center_norm[1];
            let ny = center_norm[1] * c + sign2 as f64 * h * center_norm[0];

            let tangent_1_loc = [
                round_to(start_loc[0] - start_radius * nx, 5) + 0.0,
                round_to(start_loc[1] - start_radius * ny, 5) + 0.0,
            ];
            let tangent_2_loc = [
                round_to(end_loc[0] - sign1 as f64 * end_radius * nx, 5) + 0.0,
                round_to(end_loc[1] - sign1 as f64 * end_radius * ny, 5) + 0.0,
            ];

            let tan_node_start = Node::new(tangent_1_loc, start_uuid);
            let tan_node_end = Node::new(tangent_2_loc, end_uuid);

            tangents.push((tan_node_start, tan_node_end));
        }
        if start_radius == 0.0 {
            return tangents;
        }
    }
    tangents
}

fn round_to(num: f64, places: i32) -> f64 {
    let p = 100.0_f64.powi(places);
    (num * p).round() / p
}
