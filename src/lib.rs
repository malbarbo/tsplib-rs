use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, BufRead};
use std::path::Path;
use std::str::FromStr;

macro_rules! err_invalid_data {
    ($fmt:expr, $($v:ident),*) => (
        Err(invalid_data(format!($fmt, $($v),*)))
    )
}

#[derive(Default, Debug)]
pub struct Instance {
    pub name: String,
    pub type_: Option<Type>,
    pub comment: Vec<String>,
    pub dimension: usize,
    pub capacity: usize,
    pub edge_data: Option<EdgeData>,
    pub edge_weight: Option<EdgeWeight>,
    pub edge_weight_type: Option<EdgeWeightType>,
    pub fixed_edges: Vec<(usize, usize)>,
    pub node_coord: Option<NodeCoord>,
    pub display_data: Option<Vec<(usize, f64, f64)>>,
    pub display_data_type: Option<DisplayDataType>,
    pub tour: Option<Vec<usize>>,
}

#[derive(Debug)]
pub enum EdgeData {
    EdgeList(Vec<(usize, usize)>),
    AdjList(Vec<Vec<usize>>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Type {
    Tsp,
    Atsp,
    Sop,
    Hcp,
    Cvrp,
    Tour,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EdgeWeightType {
    Explicit,
    Euc2d,
    Euc3d,
    Max2d,
    Max3d,
    Man2d,
    Man3d,
    Ceil2d,
    Geo,
    Att,
    Xray1,
    Xray2,
    Special,
}

#[derive(Debug)]
pub enum EdgeWeight {
    Function,
    FullMatrix(Vec<usize>),
    UpperRow(Vec<usize>),
    LowerRow(Vec<usize>),
    UpperDiagRow(Vec<usize>),
    LowerDiagRow(Vec<usize>),
    UpperCol(Vec<usize>),
    LowerCol(Vec<usize>),
    UpperDiagCol(Vec<usize>),
    LowerDiagCol(Vec<usize>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DisplayDataType {
    Coord,
    TwoD,
}

#[derive(Debug)]
pub enum NodeCoord {
    Two(Vec<(usize, f32, f32)>),
    Three(Vec<(usize, f32, f32, f32)>),
}

pub fn read<P: AsRef<Path>>(path: P) -> io::Result<Instance> {
    parse(BufReader::new(File::open(path)?))
}

pub fn parse<R: BufRead>(mut reader: R) -> io::Result<Instance> {
    let mut instance = Instance::default();
    let mut line = String::new();
    let mut keyword = String::new();
    let mut value = String::new();

    while reader.read_line(&mut line)? != 0 && line.trim() != EOF {
        let r = {
            let s = line.trim();
            s.is_empty() || s == "-1"
        };
        if r {
            line.clear();
            continue;
        }
        let (keyword, value) = {
            let (k, v) = split_colon(&line);
            keyword.clear();
            keyword.push_str(k);
            value.clear();
            value.push_str(v);
            (keyword.as_str(), value.as_str())
        };
        match keyword {
            NAME => {
                instance.name = value.into();
            }
            COMMENT => {
                instance.comment.push(value.into());
            }
            TYPE => {
                instance.set_type(value)?;
            }
            DIMENSION => {
                instance.dimension = parse_value(value)?;
            }
            CAPACITY => {
                instance.capacity = parse_value(value)?;
            }
            EDGE_WEIGHT_TYPE => {
                instance.set_edge_weight_type(value)?;
            }
            EDGE_WEIGHT_FORMAT => instance.set_edge_weight_format(value)?,
            EDGE_DATA_FORMAT => {
                instance.set_edge_data_format(value)?;
            }
            DISPLAY_DATA_TYPE => {
                instance.set_display_data_type(value)?;
            }
            NODE_COORD_TYPE => instance.set_node_coord_type(value)?,
            EDGE_DATA_SECTION => {
                parse_edge_data_section(&mut instance, &mut reader)?;
            }
            NODE_COORD_SECTION => {
                parse_node_coord_section(&mut instance, &mut reader)?;
            }
            EDGE_WEIGHT_SECTION => {
                parse_edge_weight_section(&mut instance, &mut reader)?;
            }
            DISPLAY_DATA_SECTION => {
                parse_display_data_section(&mut instance, &mut reader)?;
            }
            FIXED_EDGES_SECTION => {
                parse_fixed_edges_section(&mut instance, &mut reader)?;
            }
            TOUR_SECTION => {
                parse_tour_section(&mut instance, &mut reader)?;
            }
            DEPOT_SECTION | DEMAND_SECTION => {
                return err_invalid_data!("not implemented: {}", keyword)
            }
            _ => return err_invalid_data!("invalid keyword: {}", keyword),
        }
        line.clear();
    }

    Ok(instance)
}

impl Instance {
    fn num_edges(&self) -> Option<usize> {
        let m = if let Some(ref edge_data) = self.edge_data {
            use EdgeData::*;
            match *edge_data {
                EdgeList(ref v) => v.len(),
                AdjList(ref v) => v.iter().map(Vec::len).sum::<usize>() - v.len(),
            }
        } else if let Some(ref edge_weight) = self.edge_weight {
            use EdgeWeight::*;
            let n = self.dimension;
            match *edge_weight {
                FullMatrix(_) => n * n,
                UpperRow(_) | LowerRow(_) | UpperCol(_) | LowerCol(_) => (n * n - n) / 2,
                UpperDiagRow(_) | LowerDiagRow(_) | UpperDiagCol(_) | LowerDiagCol(_) => {
                    (n * n + n) / 2
                }
                _ => return None,
            }
        } else {
            return None;
        };
        Some(m)
    }

    fn set_type(&mut self, value: &str) -> io::Result<()> {
        use Type::*;
        let type_ = match value {
            ATSP => Atsp,
            CVRP => Cvrp,
            HCP => Hcp,
            SOP => Sop,
            TOUR => Tour,
            TSP | "TSP (M.~Hofmeister)" => Tsp,
            _ => return err_invalid_data!("invalid TYPE: {}", value),
        };
        self.type_ = Some(type_);
        Ok(())
    }

    fn set_edge_weight_type(&mut self, value: &str) -> io::Result<()> {
        use EdgeWeightType::*;
        let edge_weight_type = match value {
            EXPLICIT => Explicit,
            EUC_2D => Euc2d,
            EUC_3D => Euc3d,
            MAX_2D => Max2d,
            MAX_3D => Max3d,
            MAN_2D => Man2d,
            MAN_3D => Man3d,
            CEIL_2D => Ceil2d,
            GEO => Geo,
            ATT => Att,
            XRAY1 => Xray1,
            XRAY2 => Xray2,
            SPECIAL => Special,
            _ => return err_invalid_data!("invalid EDGE_WEIGHT_TYPE: {}", value),
        };
        self.edge_weight_type = Some(edge_weight_type);
        self.node_coord = match value {
            EXPLICIT => None,
            EUC_2D | MAX_2D | MAN_2D | CEIL_2D | GEO | ATT => Some(NodeCoord::Two(vec![])),
            EUC_3D | MAX_3D | MAN_3D => Some(NodeCoord::Three(vec![])),
            XRAY1 | XRAY2 | SPECIAL => return err_invalid_data!("not implemented: {}", value),
            _ => return err_invalid_data!("invalid EDGE_WEIGHT_TYPE: {}", value),
        };
        Ok(())
    }

    fn set_edge_weight_format(&mut self, value: &str) -> io::Result<()> {
        use EdgeWeight::*;
        let v = vec![];
        let edge_weight = match value {
            FUNCTION => Function,
            FULL_MATRIX => FullMatrix(v),
            UPPER_ROW => UpperRow(v),
            LOWER_ROW => LowerRow(v),
            UPPER_DIAG_ROW => UpperDiagRow(v),
            LOWER_DIAG_ROW => LowerDiagRow(v),
            UPPER_COL => UpperCol(v),
            LOWER_COL => LowerCol(v),
            UPPER_DIAG_COL => UpperDiagCol(v),
            LOWER_DIAG_COL => LowerDiagCol(v),
            _ => return err_invalid_data!("invalid EDGE_WEIGHT_FORMAT: {}", value),
        };
        self.edge_weight = Some(edge_weight);
        Ok(())
    }

    fn set_edge_data_format(&mut self, value: &str) -> io::Result<()> {
        use EdgeData::*;
        let edge_data = match value {
            EDGE_LIST => EdgeList(vec![]),
            ADJ_LIST => AdjList(vec![]),
            _ => return err_invalid_data!("invalid EDGE_DATA_FORMAT: {}", value),
        };
        self.edge_data = Some(edge_data);
        Ok(())
    }

    fn set_display_data_type(&mut self, value: &str) -> io::Result<()> {
        use DisplayDataType::*;
        self.display_data_type = match value {
            COORD_DISPLAY => Some(Coord),
            TWOD_DISPLAY => Some(TwoD),
            NO_DISPLAY => None,
            _ => return err_invalid_data!("invalid DISPLAY_DATA_TYPE: {}", value),
        };
        Ok(())
    }

    fn set_node_coord_type(&mut self, value: &str) -> io::Result<()> {
        use NodeCoord::*;
        self.node_coord = match value {
            TWOD_COORDS => Some(Two(vec![])),
            THREED_COORDS => Some(Three(vec![])),
            NO_COORDS => None,
            _ => return err_invalid_data!("invalid NODE_COORD_TYPE: {}", value),
        };
        Ok(())
    }
}


fn parse_edge_data_section<R: BufRead>(instance: &mut Instance, reader: &mut R) -> io::Result<()> {
    use EdgeData::*;
    match instance.edge_data {
        Some(EdgeList(ref mut v)) => {
            parse_vec(reader, v)?;
        }
        Some(AdjList(ref mut v)) => {
            parse_vecs(reader, v)?;
        }
        None => return Err(invalid_data("EDGE_DATA_SECTION without EDGE_DATA_FORMAT")),
    }
    Ok(())
}

fn parse_edge_weight_section<R: BufRead>(instance: &mut Instance,
                                         reader: &mut R)
                                         -> io::Result<()> {
    use EdgeWeight::*;
    let num_edges = instance.num_edges();
    match instance.edge_weight {
        Some(Function) => {
            panic!();
        }
        Some(FullMatrix(ref mut v)) |
        Some(UpperRow(ref mut v)) |
        Some(LowerRow(ref mut v)) |
        Some(UpperDiagRow(ref mut v)) |
        Some(LowerDiagRow(ref mut v)) |
        Some(UpperCol(ref mut v)) |
        Some(LowerCol(ref mut v)) |
        Some(UpperDiagCol(ref mut v)) |
        Some(LowerDiagCol(ref mut v)) => parse_all(reader, v, num_edges)?,
        None => return Err(invalid_data("EDGE_WEIGHT_FORMAT without EDGE_WEIGHT_SECTION")),
    }
    Ok(())
}

fn parse_node_coord_section<R: BufRead>(instance: &mut Instance, reader: &mut R) -> io::Result<()> {
    use NodeCoord::*;
    match instance.node_coord {
        Some(Two(ref mut v)) => {
            parse_vec(reader, v)?;
        }
        Some(Three(ref mut v)) => {
            parse_vec(reader, v)?;
        }
        None => return Err(invalid_data("EDGE_DATA_SECTION without EDGE_DATA_FORMAT")),
    }
    Ok(())
}

fn parse_display_data_section<R: BufRead>(instance: &mut Instance,
                                          reader: &mut R)
                                          -> io::Result<()> {
    if let Some(DisplayDataType::TwoD) = instance.display_data_type {
        let mut v = vec![];
        parse_vec(reader, &mut v)?;
        instance.display_data = Some(v);
    } else {
        return Err(invalid_data("DISPLAY_DATA_SECTION without EDGE_DATA_TYPE : TWOD_DISPLAY"));
    }
    Ok(())
}

fn parse_fixed_edges_section<R: BufRead>(instance: &mut Instance,
                                         reader: &mut R)
                                         -> io::Result<()> {
    let mut v = vec![];
    parse_vec(reader, &mut v)?;
    instance.fixed_edges = v;
    Ok(())
}

fn parse_tour_section<R: BufRead>(instance: &mut Instance, reader: &mut R) -> io::Result<()> {
    let mut v = vec![];
    let num = if instance.dimension == 0 {
        None
    } else {
        Some(instance.dimension)
    };
    parse_all(reader, &mut v, num)?;
    instance.tour = Some(v);
    Ok(())
}

fn parse_value<T>(s: &str) -> io::Result<T>
    where T: FromStr,
          T::Err: Into<Box<Error + Send + Sync>>
{
    s.parse().map_err(invalid_data)
}

fn parse_vec<T, R: BufRead>(reader: &mut R, v: &mut Vec<T>) -> io::Result<()>
    where T: MyFromStr
{
    let mut line = String::new();
    while reader.read_line(&mut line)? != 0 && !is_eof(&line) {
        match MyFromStr::from_str(&line) {
            Ok(a) => {
                v.push(a);
            }
            Err(a) => {
                return Err(invalid_data(a));
            }
        }
        line.clear();
    }
    Ok(())
}

fn parse_all<T, R: BufRead>(reader: &mut R, v: &mut Vec<T>, num: Option<usize>) -> io::Result<()>
    where T: FromStr + Copy,
          T::Err: Into<Box<Error + Send + Sync>>
{
    let mut line = String::new();
    while Some(v.len()) != num && reader.read_line(&mut line)? != 0 && !is_eof(&line) {
        let values: Result<Vec<T>, _> = line.split_whitespace().map(|s| s.parse::<T>()).collect();
        v.extend(values.map_err(invalid_data)?);
        line.clear();
    }
    if let Some(num) = num {
        if v.len() < num {
            return Err(invalid_data("too few values"));
        } else if v.len() > num {
            return Err(invalid_data("too many values"));
        }
    }
    Ok(())
}

fn parse_vecs<T, R: BufRead>(reader: &mut R, v: &mut Vec<Vec<T>>) -> io::Result<()>
    where T: FromStr + Copy,
          T::Err: Into<Box<Error + Send + Sync>>
{
    let mut line = String::new();
    while reader.read_line(&mut line)? != 0 && !is_eof(&line) {
        let values: Result<Vec<T>, _> = line.split_whitespace().map(|s| s.parse::<T>()).collect();
        v.push(values.map_err(invalid_data)?);
        line.clear();
    }
    Ok(())
}

fn is_eof(s: &str) -> bool {
    let s = s.trim();
    s.is_empty() || s == "-1" || s == EOF
}

fn split_colon(line: &str) -> (&str, &str) {
    if let Some(p) = line.find(':') {
        (line[..p].trim(), line[p + 1..].trim())
    } else {
        (line.trim(), &line[..0])
    }
}

fn invalid_data<E>(error: E) -> io::Error
    where E: Into<Box<Error + Send + Sync>>
{
    io::Error::new(io::ErrorKind::InvalidData, error)
}

trait MyFromStr: Sized {
    fn from_str(s: &str) -> Result<Self, Box<Error + Send + Sync>>;
}

macro_rules! impl_from_str {
    ($($T:ident),*) => (
        impl<$($T),*> MyFromStr for ($($T),*)
            where $($T: FromStr, $T::Err: Into<Box<Error + Send + Sync>>),*
        {
            #[allow(non_snake_case)]
            fn from_str(s: &str) -> Result<Self, Box<Error + Send + Sync>> {
                let mut iter = s.split_whitespace();
                $(
                    let $T;
                    if let Some(x) = iter.next() {
                        $T = x.parse().map_err(invalid_data)?;
                    } else {
                        return Err(format!("too few values: {}", s).into());
                    }
                )*

                if iter.next().is_some() {
                    return Err(format!("too many values: {}", s).into());
                }

                Ok(($($T),*))
            }
        }
    )
}

impl_from_str!(A, B);
impl_from_str!(A, B, C);
impl_from_str!(A, B, C, D);

macro_rules! def_str_consts {
    ($($name:ident),*) => (
        $(const $name: &str = stringify!($name);)*
    )
}

def_str_consts! {
    NAME,
    COMMENT,
    TYPE,
        TSP, ATSP, SOP, HCP, CVRP, TOUR,
    DIMENSION,
    CAPACITY,
    EDGE_WEIGHT_TYPE,
        EXPLICIT, EUC_2D, EUC_3D, MAX_2D, MAX_3D, MAN_2D, MAN_3D, CEIL_2D, GEO, ATT, XRAY1, XRAY2,
        SPECIAL,
    EDGE_WEIGHT_FORMAT,
        FUNCTION, FULL_MATRIX, UPPER_ROW, LOWER_ROW, UPPER_DIAG_ROW, LOWER_DIAG_ROW, UPPER_COL,
        LOWER_COL, UPPER_DIAG_COL, LOWER_DIAG_COL,
    EDGE_DATA_FORMAT,
        EDGE_LIST, ADJ_LIST,
    NODE_COORD_TYPE,
        TWOD_COORDS, THREED_COORDS, NO_COORDS,
    DISPLAY_DATA_TYPE,
        COORD_DISPLAY, TWOD_DISPLAY, NO_DISPLAY,
    DEMAND_SECTION,
    DEPOT_SECTION,
    DISPLAY_DATA_SECTION,
    EDGE_DATA_SECTION,
    EDGE_WEIGHT_SECTION,
    FIXED_EDGES_SECTION,
    NODE_COORD_SECTION,
    TOUR_SECTION,
    EOF
}
