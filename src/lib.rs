#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]
#![allow(deref_nullptr)]

// -------------------------------------------------------------------------------------------------
// LINKING
// -------------------------------------------------------------------------------------------------

#[link(name = "mediagraph")]
extern "C" {}

use opencv::prelude::*;
use std::ffi::CString;

mod bindings;

pub use bindings::*;

impl Default for Landmark {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            visibility: 0.0,
            presence: 0.0,
        }
    }
}

impl Default for Pose {
    fn default() -> Self {
        Self {
            data: [Landmark::default(); 33],
        }
    }
}

impl Default for Hand {
    fn default() -> Self {
        Self {
            data: [Landmark::default(); 21],
        }
    }
}

impl Default for FaceMesh {
    fn default() -> Self {
        Self {
            data: [Landmark::default(); 478],
        }
    }
}

pub mod pose {
    use super::*;

    pub enum PoseLandmark {
        NOSE = 0,
        LEFT_EYE_INNER = 1,
        LEFT_EYE = 2,
        LEFT_EYE_OUTER = 3,
        RIGHT_EYE_INNER = 4,
        RIGHT_EYE = 5,
        RIGHT_EYE_OUTER = 6,
        LEFT_EAR = 7,
        RIGHT_EAR = 8,
        MOUTH_LEFT = 9,
        MOUTH_RIGHT = 10,
        LEFT_SHOULDER = 11,
        RIGHT_SHOULDER = 12,
        LEFT_ELBOW = 13,
        RIGHT_ELBOW = 14,
        LEFT_WRIST = 15,
        RIGHT_WRIST = 16,
        LEFT_PINKY = 17,
        RIGHT_PINKY = 18,
        LEFT_INDEX = 19,
        RIGHT_INDEX = 20,
        LEFT_THUMB = 21,
        RIGHT_THUMB = 22,
        LEFT_HIP = 23,
        RIGHT_HIP = 24,
        LEFT_KNEE = 25,
        RIGHT_KNEE = 26,
        LEFT_ANKLE = 27,
        RIGHT_ANKLE = 28,
        LEFT_HEEL = 29,
        RIGHT_HEEL = 30,
        LEFT_FOOT_INDEX = 31,
        RIGHT_FOOT_INDEX = 32,
    }

    pub struct PoseDetector {
        pub mode: bool,         // false
        pub smooth: bool,       // true,
        pub detection_con: f32, // 0.5
        pub track_con: f32,     // 0.5
        pub graph: PoseGraph,
    }

    impl PoseDetector {
        pub fn new(mode: bool, smooth: bool, detection_con: f32, track_con: f32) -> Self {
            let graph_config =
                CString::new(include_str!("pose_tracking_cpu.txt")).expect("CString::new failed");
            let output_node = CString::new("pose_landmarks").expect("CString::new failed");

            let graph: PoseGraph =
                unsafe { PoseGraph::new(graph_config.as_ptr(), output_node.as_ptr()) };

            Self {
                mode,
                smooth,
                detection_con,
                track_con,
                graph,
            }
        }

        pub fn process(&mut self, input: &Mat, pose: *mut Pose) -> bool {
            unsafe {
                let frame = input.as_raw() as *const cv_Mat;
                self.graph.process(frame, pose)
            }
        }

        // // draw true
        // pub fn find_pose(&self, img: &[u8], draw: bool) {}

        // // draw: true, bbox_with_hands: false
        // pub fn find_position(&self, img: &[u8], draw: bool, bbox_with_hands: bool) {}

        // // draw: true
        // pub fn find_angle(
        //     &self,
        //     img: &[u8],
        //     p1: cgmath::Point2<f32>,
        //     p2: cgmath::Point2<f32>,
        //     draw: bool,
        // ) {
        // }

        // pub fn find_distance(
        //     &self,
        //     p1: cgmath::Point2<f32>,
        //     p2: cgmath::Point2<f32>,
        //     img: Option<&[u8]>,
        //     r: f32,
        //     t: f32,
        // ) {
        // }

        // pub fn anlge_check(&self, my_angle: f32, target_angle: f32, add_on: f32) {}
    }

    impl Default for PoseDetector {
        fn default() -> Self {
            Self::new(false, true, 0.5, 0.5)
        }
    }
}

pub mod face_mesh {
    use super::*;

    pub struct FaceMeshDetector {
        pub static_mode: bool,      // false
        pub max_faces: usize,       // 2
        pub min_detection_con: f32, // 0.5
        pub min_track_con: f32,     // 0.5
        pub graph: FaceMeshGraph,
    }

    impl FaceMeshDetector {
        pub fn new(
            static_mode: bool,
            max_faces: usize,
            min_detection_con: f32,
            min_track_con: f32,
        ) -> Self {
            let graph_config = CString::new(include_str!("face_mesh_desktop_live.txt"))
                .expect("CString::new failed");
            let output_node = CString::new("multi_face_landmarks").expect("CString::new failed");

            let graph: FaceMeshGraph =
                unsafe { FaceMeshGraph::new(graph_config.as_ptr(), output_node.as_ptr()) };
            Self {
                static_mode,
                max_faces,
                min_detection_con,
                min_track_con,
                graph,
            }
        }

        pub fn process(&mut self, input: &Mat, mesh: *mut FaceMesh) -> bool {
            unsafe {
                let frame = input.as_raw() as *const cv_Mat;
                self.graph.process(frame, mesh)
            }
        }
        // // draw: true
        // pub fn find_face_mesh(&self, img: &[u8], draw: bool) {}

        // pub fn find_distance(
        //     &self,
        //     p1: cgmath::Point2<f32>,
        //     p2: cgmath::Point2<f32>,
        //     img: Option<&[u8]>,
        // ) {
        // }
    }

    impl Default for FaceMeshDetector {
        fn default() -> Self {
            Self::new(false, 2, 0.5, 0.5)
        }
    }
}

// pub mod face_detection {
//     pub enum FaceKeyPoint {
//         RIGHT_EYE = 0,
//         LEFT_EYE = 1,
//         NOSE_TIP = 2,
//         MOUTH_CENTER = 3,
//         RIGHT_EAR_TRAGION = 4,
//         LEFT_EAR_TRAGION = 5,
//     }
//     pub struct FaceDetection {}

//     impl FaceDetection {
//         pub fn process(&self /* image */) /*NamedTuple*/ {}
//     }

//     pub struct FaceDetector {
//         pub min_detection_con: f32, // 0.5
//         pub face_detection: FaceDetection,
//     }

//     impl FaceDetector {
//         pub fn new(min_detection_con: f32) -> Self {
//             Self {
//                 min_detection_con,
//                 face_detection: todo!(),
//             }
//         }

//         // draw: true
//         pub fn find_faces(&self, img: &[u8], draw: bool) {}
//     }

//     impl Default for FaceDetector {
//         fn default() -> Self {
//             Self::new(0.5)
//         }
//     }
// }

pub mod hands {
    use super::*;
    // use mediapipe::*;
    // use std::collections::HashMap;

    pub enum HandLandmark {
        WRIST = 0,
        THUMB_CMC = 1,
        THUMB_MCP = 2,
        THUMB_IP = 3,
        THUMB_TIP = 4,
        INDEX_FINGER_MCP = 5,
        INDEX_FINGER_PIP = 6,
        INDEX_FINGER_DIP = 7,
        INDEX_FINGER_TIP = 8,
        MIDDLE_FINGER_MCP = 9,
        MIDDLE_FINGER_PIP = 10,
        MIDDLE_FINGER_DIP = 11,
        MIDDLE_FINGER_TIP = 12,
        RING_FINGER_MCP = 13,
        RING_FINGER_PIP = 14,
        RING_FINGER_DIP = 15,
        RING_FINGER_TIP = 16,
        PINKY_MCP = 17,
        PINKY_PIP = 18,
        PINKY_DIP = 19,
        PINKY_TIP = 20,
    }

    pub struct HandDetector {
        pub mode: bool,
        pub max_hands: usize,
        pub detection_con: f32, // 0.5
        pub min_track_con: f32, // 0.5
        pub graph: HandsGraph,
    }

    impl HandDetector {
        pub fn new(mode: bool, max_hands: usize, detection_con: f32, min_track_con: f32) -> Self {
            // // ::std::vector<::mediapipe::NormalizedLandmarkList>
            // let graph_config = CString::new(include_str!("face_mesh_desktop_live.txt")).expect("CString::new failed");
            // let output_node = CString::new("multi_face_landmarks").expect("CString::new failed");

            let graph_config = CString::new(include_str!("hand_tracking_desktop_live.txt"))
                .expect("CString::new failed");
            let output_node = CString::new("hand_landmarks").expect("CString::new failed");

            let graph: HandsGraph =
                unsafe { HandsGraph::new(graph_config.as_ptr(), output_node.as_ptr()) };

            Self {
                mode,
                max_hands,
                detection_con,
                min_track_con,
                graph,
            }
        }

        pub fn process(&mut self, input: &Mat, left: *mut Hand, right: *mut Hand) -> bool {
            unsafe {
                let frame = input.as_raw() as *const cv_Mat;
                self.graph.process(frame, left, right)
            }
        }
        // // draw: true, flip_type: tru
        // pub fn find_hands(&self, img: &[u8], draw: bool, flip_type: bool) {}

        // pub fn fingers_up(&self, my_hand: &HashMap<String, String>) /*List of which fingers are up*/
        // {
        // }

        // pub fn find_distance(
        //     &self,
        //     p1: cgmath::Point2<f32>,
        //     p2: cgmath::Point2<f32>,
        //     img: Option<&[u8]>,
        // ) {
        // }
    }

    impl Default for HandDetector {
        fn default() -> Self {
            Self::new(false, 2, 0.5, 0.5)
        }
    }
}

// pub mod objectron {
//     pub struct Objectron {}

//     impl Objectron {
//         pub fn process(&self /* image */) /*NamedTuple*/ {}
//     }
// }

// pub mod selfie_segmentation {
//     pub struct SelfieSegmentation {}

//     impl SelfieSegmentation {
//         pub fn process(&self /* image */) /*NamedTuple*/ {}
//     }

//     pub struct SelfieSegmentationDetector {
//         pub model: usize, // 0 is general 1 is landscape(faster)
//         pub selfie_segmentation: SelfieSegmentation,
//     }

//     impl SelfieSegmentationDetector {
//         pub fn new(model: usize) -> Self {
//             todo!()
//         }

//         // threshold: 0.1
//         pub fn remove_bg(&self, img: &[u8], img_bg: [u8; 3], threshold: f32) {}
//     }

//     impl Default for SelfieSegmentationDetector {
//         fn default() -> Self {
//             Self::new(1)
//         }
//     }
// }

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
