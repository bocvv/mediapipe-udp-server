// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include <iostream>
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/wrapper_iris.pb.h"


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#define PORT    8765
#define MAXLINE 1024
int sockfd;
struct sockaddr_in     servaddr;

namespace mediapipe {

constexpr char kEyeLandmarksLeftTag[] = "EYE_LANDMARKS_LEFT";
constexpr char kEyeLandmarksRightTag[] = "EYE_LANDMARKS_RIGHT";
constexpr char kIrisLandmarksLeftTag[] = "IRIS_LANDMARKS_LEFT";
constexpr char kIrisLandmarksRightTag[] = "IRIS_LANDMARKS_RIGHT";
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS"; // MAD note @to-fix: streaming NORM_LANDMARKS, but they're labeled LANDMARKS
static int frame_id = 0;


void setup_udp(){
  // int sockfd;
  char buffer[MAXLINE];
  // char *hello = "Hello from client";
  // struct sockaddr_in     servaddr;

  // Creating socket file descriptor
  if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) {
      perror("socket creation failed");
      exit(EXIT_FAILURE);
  }

  memset(&servaddr, 0, sizeof(servaddr));

  // Filling server information
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(PORT);
  servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  
  // initial testing for connection
  // while(true){
  //   sendto(sockfd, (const char *)hello, strlen(hello),
  //       0, (const struct sockaddr *) &servaddr,
  //           sizeof(servaddr));
  //   printf("Hello message sent.\n");
  // }
  // close(sockfd);
}


// A Calculator that simply passes its input Packets and header through,
// unchanged.  The inputs may be specified by tag or index.  The outputs
// must match the inputs exactly.  Any number of input side packets may
// also be specified.  If output side packets are specified, they must
// match the input side packets exactly and the Calculator passes its
// input side packets through, unchanged.  Otherwise, the input side
// packets will be ignored (allowing PassThroughCalculator to be used to
// test internal behavior).  Any options may be specified and will be
// ignored.
class IrisPassThroughCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    if (!cc->Inputs().TagMap()->SameAs(*cc->Outputs().TagMap())) {
      return ::mediapipe::InvalidArgumentError(
          "Input and output streams to PassThroughCalculator must use "
          "matching tags and indexes.");
    }
    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      cc->Inputs().Get(id).SetAny();
      cc->Outputs().Get(id).SetSameAs(&cc->Inputs().Get(id));
    }
    for (CollectionItemId id = cc->InputSidePackets().BeginId();
         id < cc->InputSidePackets().EndId(); ++id) {
      cc->InputSidePackets().Get(id).SetAny();
    }
    if (cc->OutputSidePackets().NumEntries() != 0) {
      if (!cc->InputSidePackets().TagMap()->SameAs(
              *cc->OutputSidePackets().TagMap())) {
        return ::mediapipe::InvalidArgumentError(
            "Input and output side packets to PassThroughCalculator must use "
            "matching tags and indexes.");
      }
      for (CollectionItemId id = cc->InputSidePackets().BeginId();
           id < cc->InputSidePackets().EndId(); ++id) {
        cc->OutputSidePackets().Get(id).SetSameAs(
            &cc->InputSidePackets().Get(id));
      }
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      if (!cc->Inputs().Get(id).Header().IsEmpty()) {
        cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
      }
    }
    if (cc->OutputSidePackets().NumEntries() != 0) {
      for (CollectionItemId id = cc->InputSidePackets().BeginId();
           id < cc->InputSidePackets().EndId(); ++id) {
        cc->OutputSidePackets().Get(id).Set(cc->InputSidePackets().Get(id));
      }
    }
    cc->SetOffset(TimestampDiff(0));


    setup_udp();


    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    cc->GetCounter("PassThrough")->Increment();
    if (cc->Inputs().NumEntries() == 0) {
      return tool::StatusStop();
    }

    WrapperIris* wrapper = new WrapperIris();
    wrapper->InitAsDefaultInstance();
    // wrapper->set_frame_id(++frame_id);

    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      if (!cc->Inputs().Get(id).IsEmpty()) {


        /*-------------------------------------------------------------------*/
        /*------------ EDITS to original pass_through_calculator ------------*/
        /*-------------------------------------------------------------------*/


        if (cc->Inputs().Get(id).Name() == "left_eye_contour_landmarks") {
            // the type is a NormalizedLandmarkList, but you need the kLandmarksTag
            // in order for it not to crash for some reason ...
            const NormalizedLandmarkList& landmarks_0 = cc->Inputs().Tag(kEyeLandmarksLeftTag).Get<NormalizedLandmarkList>();
            for (int i = 0; i < landmarks_0.landmark_size(); ++i) {
                const NormalizedLandmark& landmark = landmarks_0.landmark(i);

                wrapper->mutable_eye_landmarks_left()->add_landmark();
                int size = wrapper->mutable_eye_landmarks_left()->landmark_size()-1;
                wrapper->mutable_eye_landmarks_left()->mutable_landmark(size)->set_x(landmark.x());
                wrapper->mutable_eye_landmarks_left()->mutable_landmark(size)->set_y(landmark.y());
                wrapper->mutable_eye_landmarks_left()->mutable_landmark(size)->set_z(landmark.z());
            }

            const NormalizedLandmarkList& landmarks_1 = cc->Inputs().Tag(kEyeLandmarksRightTag).Get<NormalizedLandmarkList>();
            for (int i = 0; i < landmarks_1.landmark_size(); ++i) {
                const NormalizedLandmark& landmark = landmarks_1.landmark(i);

                wrapper->mutable_eye_landmarks_right()->add_landmark();
                int size = wrapper->mutable_eye_landmarks_right()->landmark_size()-1;
                wrapper->mutable_eye_landmarks_right()->mutable_landmark(size)->set_x(landmark.x());
                wrapper->mutable_eye_landmarks_right()->mutable_landmark(size)->set_y(landmark.y());
                wrapper->mutable_eye_landmarks_right()->mutable_landmark(size)->set_z(landmark.z());
            }

            const NormalizedLandmarkList& landmarks_2 = cc->Inputs().Tag(kIrisLandmarksLeftTag).Get<NormalizedLandmarkList>();
            for (int i = 0; i < landmarks_2.landmark_size(); ++i) {
                const NormalizedLandmark& landmark = landmarks_2.landmark(i);

                wrapper->mutable_iris_landmarks_left()->add_landmark();
                int size = wrapper->mutable_iris_landmarks_left()->landmark_size()-1;
                wrapper->mutable_iris_landmarks_left()->mutable_landmark(size)->set_x(landmark.x());
                wrapper->mutable_iris_landmarks_left()->mutable_landmark(size)->set_y(landmark.y());
                wrapper->mutable_iris_landmarks_left()->mutable_landmark(size)->set_z(landmark.z());
            }

            const NormalizedLandmarkList& landmarks_3 = cc->Inputs().Tag(kIrisLandmarksRightTag).Get<NormalizedLandmarkList>();
            for (int i = 0; i < landmarks_3.landmark_size(); ++i) {
                const NormalizedLandmark& landmark = landmarks_3.landmark(i);

                wrapper->mutable_iris_landmarks_right()->add_landmark();
                int size = wrapper->mutable_iris_landmarks_right()->landmark_size()-1;
                wrapper->mutable_iris_landmarks_right()->mutable_landmark(size)->set_x(landmark.x());
                wrapper->mutable_iris_landmarks_right()->mutable_landmark(size)->set_y(landmark.y());
                wrapper->mutable_iris_landmarks_right()->mutable_landmark(size)->set_z(landmark.z());
            }

       }

        /*-------------------------------------------------------------------*/

        VLOG(3) << "Passing " << cc->Inputs().Get(id).Name() << " to "
                << cc->Outputs().Get(id).Name() << " at "
                << cc->InputTimestamp().DebugString();
        cc->Outputs().Get(id).AddPacket(cc->Inputs().Get(id).Value());
      }
    }

    std::string msg_buffer;
    wrapper->SerializeToString(&msg_buffer);

    sendto(sockfd, msg_buffer.c_str(), msg_buffer.length(),
        0, (const struct sockaddr *) &servaddr,
            sizeof(servaddr));

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Close(CalculatorContext* cc) {
    if (!cc->GraphStatus().ok()) {
      return ::mediapipe::OkStatus();
    }
    close(sockfd);
    return ::mediapipe::OkStatus();
  }

};
REGISTER_CALCULATOR(IrisPassThroughCalculator);

}  // namespace mediapipe
