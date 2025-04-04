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

class pose_communication_cameraRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.image_data = null;
    }
    else {
      if (initObj.hasOwnProperty('image_data')) {
        this.image_data = initObj.image_data
      }
      else {
        this.image_data = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type pose_communication_cameraRequest
    // Serialize message field [image_data]
    bufferOffset = _arraySerializer.float32(obj.image_data, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type pose_communication_cameraRequest
    let len;
    let data = new pose_communication_cameraRequest(null);
    // Deserialize message field [image_data]
    data.image_data = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.image_data.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'local_inn/pose_communication_cameraRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '1b1fb4ce4406affe0886a88095de24da';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Request
    float32[] image_data  # camera scan data but using only pose as of now (array of floats)
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new pose_communication_cameraRequest(null);
    if (msg.image_data !== undefined) {
      resolved.image_data = msg.image_data;
    }
    else {
      resolved.image_data = []
    }

    return resolved;
    }
};

class pose_communication_cameraResponse {
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
    // Serializes a message object of type pose_communication_cameraResponse
    // Check that the constant length array field [pose] has the right length
    if (obj.pose.length !== 3) {
      throw new Error('Unable to serialize array field pose - length must be 3')
    }
    // Serialize message field [pose]
    bufferOffset = _arraySerializer.float32(obj.pose, buffer, bufferOffset, 3);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type pose_communication_cameraResponse
    let len;
    let data = new pose_communication_cameraResponse(null);
    // Deserialize message field [pose]
    data.pose = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    return data;
  }

  static getMessageSize(object) {
    return 12;
  }

  static datatype() {
    // Returns string type for a service object
    return 'local_inn/pose_communication_cameraResponse';
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
    const resolved = new pose_communication_cameraResponse(null);
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
  Request: pose_communication_cameraRequest,
  Response: pose_communication_cameraResponse,
  md5sum() { return 'c4fc5fe5d8e49992c022543ae4cff221'; },
  datatype() { return 'local_inn/pose_communication_camera'; }
};
