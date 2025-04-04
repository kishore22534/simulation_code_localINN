// Auto-generated. Do not edit!

// (in-package local_inn.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class pose_communicationRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.lidar_data = null;
    }
    else {
      if (initObj.hasOwnProperty('lidar_data')) {
        this.lidar_data = initObj.lidar_data
      }
      else {
        this.lidar_data = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type pose_communicationRequest
    // Serialize message field [lidar_data]
    bufferOffset = _arraySerializer.float32(obj.lidar_data, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type pose_communicationRequest
    let len;
    let data = new pose_communicationRequest(null);
    // Deserialize message field [lidar_data]
    data.lidar_data = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.lidar_data.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'local_inn/pose_communicationRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '8044b6ad75eddb910fb8489678496d7f';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Request
    float32[] lidar_data  # LIDAR scan data (array of floats)
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new pose_communicationRequest(null);
    if (msg.lidar_data !== undefined) {
      resolved.lidar_data = msg.lidar_data;
    }
    else {
      resolved.lidar_data = []
    }

    return resolved;
    }
};

class pose_communicationResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.pose = null;
    }
    else {
      if (initObj.hasOwnProperty('pose')) {
        this.pose = initObj.pose
      }
      else {
        this.pose = new Array(3).fill(0);
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type pose_communicationResponse
    // Check that the constant length array field [pose] has the right length
    if (obj.pose.length !== 3) {
      throw new Error('Unable to serialize array field pose - length must be 3')
    }
    // Serialize message field [pose]
    bufferOffset = _arraySerializer.float32(obj.pose, buffer, bufferOffset, 3);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type pose_communicationResponse
    let len;
    let data = new pose_communicationResponse(null);
    // Deserialize message field [pose]
    data.pose = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    return data;
  }

  static getMessageSize(object) {
    return 12;
  }

  static datatype() {
    // Returns string type for a service object
    return 'local_inn/pose_communicationResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '5e7af384a7bc934f230231c255c5249e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Response
    float32[3] pose  # x, y, theta
    
    
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new pose_communicationResponse(null);
    if (msg.pose !== undefined) {
      resolved.pose = msg.pose;
    }
    else {
      resolved.pose = new Array(3).fill(0)
    }

    return resolved;
    }
};

module.exports = {
  Request: pose_communicationRequest,
  Response: pose_communicationResponse,
  md5sum() { return '5aa28f834219f856593a6cf0bb2c70d5'; },
  datatype() { return 'local_inn/pose_communication'; }
};
