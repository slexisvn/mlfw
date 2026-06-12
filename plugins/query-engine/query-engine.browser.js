var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __esm = (fn, res) => function __init() {
  return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
};
var __commonJS = (cb, mod) => function __require() {
  return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
};
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));

// node_modules/base64-js/index.js
var require_base64_js = __commonJS({
  "node_modules/base64-js/index.js"(exports) {
    "use strict";
    init_buffer_shim();
    exports.byteLength = byteLength;
    exports.toByteArray = toByteArray;
    exports.fromByteArray = fromByteArray;
    var lookup = [];
    var revLookup = [];
    var Arr = typeof Uint8Array !== "undefined" ? Uint8Array : Array;
    var code = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (i = 0, len = code.length; i < len; ++i) {
      lookup[i] = code[i];
      revLookup[code.charCodeAt(i)] = i;
    }
    var i;
    var len;
    revLookup["-".charCodeAt(0)] = 62;
    revLookup["_".charCodeAt(0)] = 63;
    function getLens(b64) {
      var len2 = b64.length;
      if (len2 % 4 > 0) {
        throw new Error("Invalid string. Length must be a multiple of 4");
      }
      var validLen = b64.indexOf("=");
      if (validLen === -1) validLen = len2;
      var placeHoldersLen = validLen === len2 ? 0 : 4 - validLen % 4;
      return [validLen, placeHoldersLen];
    }
    function byteLength(b64) {
      var lens = getLens(b64);
      var validLen = lens[0];
      var placeHoldersLen = lens[1];
      return (validLen + placeHoldersLen) * 3 / 4 - placeHoldersLen;
    }
    function _byteLength(b64, validLen, placeHoldersLen) {
      return (validLen + placeHoldersLen) * 3 / 4 - placeHoldersLen;
    }
    function toByteArray(b64) {
      var tmp;
      var lens = getLens(b64);
      var validLen = lens[0];
      var placeHoldersLen = lens[1];
      var arr = new Arr(_byteLength(b64, validLen, placeHoldersLen));
      var curByte = 0;
      var len2 = placeHoldersLen > 0 ? validLen - 4 : validLen;
      var i2;
      for (i2 = 0; i2 < len2; i2 += 4) {
        tmp = revLookup[b64.charCodeAt(i2)] << 18 | revLookup[b64.charCodeAt(i2 + 1)] << 12 | revLookup[b64.charCodeAt(i2 + 2)] << 6 | revLookup[b64.charCodeAt(i2 + 3)];
        arr[curByte++] = tmp >> 16 & 255;
        arr[curByte++] = tmp >> 8 & 255;
        arr[curByte++] = tmp & 255;
      }
      if (placeHoldersLen === 2) {
        tmp = revLookup[b64.charCodeAt(i2)] << 2 | revLookup[b64.charCodeAt(i2 + 1)] >> 4;
        arr[curByte++] = tmp & 255;
      }
      if (placeHoldersLen === 1) {
        tmp = revLookup[b64.charCodeAt(i2)] << 10 | revLookup[b64.charCodeAt(i2 + 1)] << 4 | revLookup[b64.charCodeAt(i2 + 2)] >> 2;
        arr[curByte++] = tmp >> 8 & 255;
        arr[curByte++] = tmp & 255;
      }
      return arr;
    }
    function tripletToBase64(num) {
      return lookup[num >> 18 & 63] + lookup[num >> 12 & 63] + lookup[num >> 6 & 63] + lookup[num & 63];
    }
    function encodeChunk(uint8, start, end) {
      var tmp;
      var output = [];
      for (var i2 = start; i2 < end; i2 += 3) {
        tmp = (uint8[i2] << 16 & 16711680) + (uint8[i2 + 1] << 8 & 65280) + (uint8[i2 + 2] & 255);
        output.push(tripletToBase64(tmp));
      }
      return output.join("");
    }
    function fromByteArray(uint8) {
      var tmp;
      var len2 = uint8.length;
      var extraBytes = len2 % 3;
      var parts = [];
      var maxChunkLength = 16383;
      for (var i2 = 0, len22 = len2 - extraBytes; i2 < len22; i2 += maxChunkLength) {
        parts.push(encodeChunk(uint8, i2, i2 + maxChunkLength > len22 ? len22 : i2 + maxChunkLength));
      }
      if (extraBytes === 1) {
        tmp = uint8[len2 - 1];
        parts.push(
          lookup[tmp >> 2] + lookup[tmp << 4 & 63] + "=="
        );
      } else if (extraBytes === 2) {
        tmp = (uint8[len2 - 2] << 8) + uint8[len2 - 1];
        parts.push(
          lookup[tmp >> 10] + lookup[tmp >> 4 & 63] + lookup[tmp << 2 & 63] + "="
        );
      }
      return parts.join("");
    }
  }
});

// node_modules/ieee754/index.js
var require_ieee754 = __commonJS({
  "node_modules/ieee754/index.js"(exports) {
    init_buffer_shim();
    exports.read = function(buffer, offset, isLE, mLen, nBytes) {
      var e, m;
      var eLen = nBytes * 8 - mLen - 1;
      var eMax = (1 << eLen) - 1;
      var eBias = eMax >> 1;
      var nBits = -7;
      var i = isLE ? nBytes - 1 : 0;
      var d = isLE ? -1 : 1;
      var s = buffer[offset + i];
      i += d;
      e = s & (1 << -nBits) - 1;
      s >>= -nBits;
      nBits += eLen;
      for (; nBits > 0; e = e * 256 + buffer[offset + i], i += d, nBits -= 8) {
      }
      m = e & (1 << -nBits) - 1;
      e >>= -nBits;
      nBits += mLen;
      for (; nBits > 0; m = m * 256 + buffer[offset + i], i += d, nBits -= 8) {
      }
      if (e === 0) {
        e = 1 - eBias;
      } else if (e === eMax) {
        return m ? NaN : (s ? -1 : 1) * Infinity;
      } else {
        m = m + Math.pow(2, mLen);
        e = e - eBias;
      }
      return (s ? -1 : 1) * m * Math.pow(2, e - mLen);
    };
    exports.write = function(buffer, value, offset, isLE, mLen, nBytes) {
      var e, m, c;
      var eLen = nBytes * 8 - mLen - 1;
      var eMax = (1 << eLen) - 1;
      var eBias = eMax >> 1;
      var rt = mLen === 23 ? Math.pow(2, -24) - Math.pow(2, -77) : 0;
      var i = isLE ? 0 : nBytes - 1;
      var d = isLE ? 1 : -1;
      var s = value < 0 || value === 0 && 1 / value < 0 ? 1 : 0;
      value = Math.abs(value);
      if (isNaN(value) || value === Infinity) {
        m = isNaN(value) ? 1 : 0;
        e = eMax;
      } else {
        e = Math.floor(Math.log(value) / Math.LN2);
        if (value * (c = Math.pow(2, -e)) < 1) {
          e--;
          c *= 2;
        }
        if (e + eBias >= 1) {
          value += rt / c;
        } else {
          value += rt * Math.pow(2, 1 - eBias);
        }
        if (value * c >= 2) {
          e++;
          c /= 2;
        }
        if (e + eBias >= eMax) {
          m = 0;
          e = eMax;
        } else if (e + eBias >= 1) {
          m = (value * c - 1) * Math.pow(2, mLen);
          e = e + eBias;
        } else {
          m = value * Math.pow(2, eBias - 1) * Math.pow(2, mLen);
          e = 0;
        }
      }
      for (; mLen >= 8; buffer[offset + i] = m & 255, i += d, m /= 256, mLen -= 8) {
      }
      e = e << mLen | m;
      eLen += mLen;
      for (; eLen > 0; buffer[offset + i] = e & 255, i += d, e /= 256, eLen -= 8) {
      }
      buffer[offset + i - d] |= s * 128;
    };
  }
});

// node_modules/buffer/index.js
var require_buffer = __commonJS({
  "node_modules/buffer/index.js"(exports) {
    "use strict";
    init_buffer_shim();
    var base64 = require_base64_js();
    var ieee754 = require_ieee754();
    var customInspectSymbol = typeof Symbol === "function" && typeof Symbol["for"] === "function" ? Symbol["for"]("nodejs.util.inspect.custom") : null;
    exports.Buffer = Buffer3;
    exports.SlowBuffer = SlowBuffer;
    exports.INSPECT_MAX_BYTES = 50;
    var K_MAX_LENGTH = 2147483647;
    exports.kMaxLength = K_MAX_LENGTH;
    Buffer3.TYPED_ARRAY_SUPPORT = typedArraySupport();
    if (!Buffer3.TYPED_ARRAY_SUPPORT && typeof console !== "undefined" && typeof console.error === "function") {
      console.error(
        "This browser lacks typed array (Uint8Array) support which is required by `buffer` v5.x. Use `buffer` v4.x if you require old browser support."
      );
    }
    function typedArraySupport() {
      try {
        const arr = new Uint8Array(1);
        const proto = { foo: function() {
          return 42;
        } };
        Object.setPrototypeOf(proto, Uint8Array.prototype);
        Object.setPrototypeOf(arr, proto);
        return arr.foo() === 42;
      } catch (e) {
        return false;
      }
    }
    Object.defineProperty(Buffer3.prototype, "parent", {
      enumerable: true,
      get: function() {
        if (!Buffer3.isBuffer(this)) return void 0;
        return this.buffer;
      }
    });
    Object.defineProperty(Buffer3.prototype, "offset", {
      enumerable: true,
      get: function() {
        if (!Buffer3.isBuffer(this)) return void 0;
        return this.byteOffset;
      }
    });
    function createBuffer(length) {
      if (length > K_MAX_LENGTH) {
        throw new RangeError('The value "' + length + '" is invalid for option "size"');
      }
      const buf = new Uint8Array(length);
      Object.setPrototypeOf(buf, Buffer3.prototype);
      return buf;
    }
    function Buffer3(arg, encodingOrOffset, length) {
      if (typeof arg === "number") {
        if (typeof encodingOrOffset === "string") {
          throw new TypeError(
            'The "string" argument must be of type string. Received type number'
          );
        }
        return allocUnsafe(arg);
      }
      return from(arg, encodingOrOffset, length);
    }
    Buffer3.poolSize = 8192;
    function from(value, encodingOrOffset, length) {
      if (typeof value === "string") {
        return fromString(value, encodingOrOffset);
      }
      if (ArrayBuffer.isView(value)) {
        return fromArrayView(value);
      }
      if (value == null) {
        throw new TypeError(
          "The first argument must be one of type string, Buffer, ArrayBuffer, Array, or Array-like Object. Received type " + typeof value
        );
      }
      if (isInstance(value, ArrayBuffer) || value && isInstance(value.buffer, ArrayBuffer)) {
        return fromArrayBuffer(value, encodingOrOffset, length);
      }
      if (typeof SharedArrayBuffer !== "undefined" && (isInstance(value, SharedArrayBuffer) || value && isInstance(value.buffer, SharedArrayBuffer))) {
        return fromArrayBuffer(value, encodingOrOffset, length);
      }
      if (typeof value === "number") {
        throw new TypeError(
          'The "value" argument must not be of type number. Received type number'
        );
      }
      const valueOf = value.valueOf && value.valueOf();
      if (valueOf != null && valueOf !== value) {
        return Buffer3.from(valueOf, encodingOrOffset, length);
      }
      const b = fromObject(value);
      if (b) return b;
      if (typeof Symbol !== "undefined" && Symbol.toPrimitive != null && typeof value[Symbol.toPrimitive] === "function") {
        return Buffer3.from(value[Symbol.toPrimitive]("string"), encodingOrOffset, length);
      }
      throw new TypeError(
        "The first argument must be one of type string, Buffer, ArrayBuffer, Array, or Array-like Object. Received type " + typeof value
      );
    }
    Buffer3.from = function(value, encodingOrOffset, length) {
      return from(value, encodingOrOffset, length);
    };
    Object.setPrototypeOf(Buffer3.prototype, Uint8Array.prototype);
    Object.setPrototypeOf(Buffer3, Uint8Array);
    function assertSize(size) {
      if (typeof size !== "number") {
        throw new TypeError('"size" argument must be of type number');
      } else if (size < 0) {
        throw new RangeError('The value "' + size + '" is invalid for option "size"');
      }
    }
    function alloc(size, fill, encoding) {
      assertSize(size);
      if (size <= 0) {
        return createBuffer(size);
      }
      if (fill !== void 0) {
        return typeof encoding === "string" ? createBuffer(size).fill(fill, encoding) : createBuffer(size).fill(fill);
      }
      return createBuffer(size);
    }
    Buffer3.alloc = function(size, fill, encoding) {
      return alloc(size, fill, encoding);
    };
    function allocUnsafe(size) {
      assertSize(size);
      return createBuffer(size < 0 ? 0 : checked(size) | 0);
    }
    Buffer3.allocUnsafe = function(size) {
      return allocUnsafe(size);
    };
    Buffer3.allocUnsafeSlow = function(size) {
      return allocUnsafe(size);
    };
    function fromString(string, encoding) {
      if (typeof encoding !== "string" || encoding === "") {
        encoding = "utf8";
      }
      if (!Buffer3.isEncoding(encoding)) {
        throw new TypeError("Unknown encoding: " + encoding);
      }
      const length = byteLength(string, encoding) | 0;
      let buf = createBuffer(length);
      const actual = buf.write(string, encoding);
      if (actual !== length) {
        buf = buf.slice(0, actual);
      }
      return buf;
    }
    function fromArrayLike(array) {
      const length = array.length < 0 ? 0 : checked(array.length) | 0;
      const buf = createBuffer(length);
      for (let i = 0; i < length; i += 1) {
        buf[i] = array[i] & 255;
      }
      return buf;
    }
    function fromArrayView(arrayView) {
      if (isInstance(arrayView, Uint8Array)) {
        const copy = new Uint8Array(arrayView);
        return fromArrayBuffer(copy.buffer, copy.byteOffset, copy.byteLength);
      }
      return fromArrayLike(arrayView);
    }
    function fromArrayBuffer(array, byteOffset, length) {
      if (byteOffset < 0 || array.byteLength < byteOffset) {
        throw new RangeError('"offset" is outside of buffer bounds');
      }
      if (array.byteLength < byteOffset + (length || 0)) {
        throw new RangeError('"length" is outside of buffer bounds');
      }
      let buf;
      if (byteOffset === void 0 && length === void 0) {
        buf = new Uint8Array(array);
      } else if (length === void 0) {
        buf = new Uint8Array(array, byteOffset);
      } else {
        buf = new Uint8Array(array, byteOffset, length);
      }
      Object.setPrototypeOf(buf, Buffer3.prototype);
      return buf;
    }
    function fromObject(obj) {
      if (Buffer3.isBuffer(obj)) {
        const len = checked(obj.length) | 0;
        const buf = createBuffer(len);
        if (buf.length === 0) {
          return buf;
        }
        obj.copy(buf, 0, 0, len);
        return buf;
      }
      if (obj.length !== void 0) {
        if (typeof obj.length !== "number" || numberIsNaN(obj.length)) {
          return createBuffer(0);
        }
        return fromArrayLike(obj);
      }
      if (obj.type === "Buffer" && Array.isArray(obj.data)) {
        return fromArrayLike(obj.data);
      }
    }
    function checked(length) {
      if (length >= K_MAX_LENGTH) {
        throw new RangeError("Attempt to allocate Buffer larger than maximum size: 0x" + K_MAX_LENGTH.toString(16) + " bytes");
      }
      return length | 0;
    }
    function SlowBuffer(length) {
      if (+length != length) {
        length = 0;
      }
      return Buffer3.alloc(+length);
    }
    Buffer3.isBuffer = function isBuffer(b) {
      return b != null && b._isBuffer === true && b !== Buffer3.prototype;
    };
    Buffer3.compare = function compare(a, b) {
      if (isInstance(a, Uint8Array)) a = Buffer3.from(a, a.offset, a.byteLength);
      if (isInstance(b, Uint8Array)) b = Buffer3.from(b, b.offset, b.byteLength);
      if (!Buffer3.isBuffer(a) || !Buffer3.isBuffer(b)) {
        throw new TypeError(
          'The "buf1", "buf2" arguments must be one of type Buffer or Uint8Array'
        );
      }
      if (a === b) return 0;
      let x = a.length;
      let y = b.length;
      for (let i = 0, len = Math.min(x, y); i < len; ++i) {
        if (a[i] !== b[i]) {
          x = a[i];
          y = b[i];
          break;
        }
      }
      if (x < y) return -1;
      if (y < x) return 1;
      return 0;
    };
    Buffer3.isEncoding = function isEncoding(encoding) {
      switch (String(encoding).toLowerCase()) {
        case "hex":
        case "utf8":
        case "utf-8":
        case "ascii":
        case "latin1":
        case "binary":
        case "base64":
        case "ucs2":
        case "ucs-2":
        case "utf16le":
        case "utf-16le":
          return true;
        default:
          return false;
      }
    };
    Buffer3.concat = function concat(list, length) {
      if (!Array.isArray(list)) {
        throw new TypeError('"list" argument must be an Array of Buffers');
      }
      if (list.length === 0) {
        return Buffer3.alloc(0);
      }
      let i;
      if (length === void 0) {
        length = 0;
        for (i = 0; i < list.length; ++i) {
          length += list[i].length;
        }
      }
      const buffer = Buffer3.allocUnsafe(length);
      let pos = 0;
      for (i = 0; i < list.length; ++i) {
        let buf = list[i];
        if (isInstance(buf, Uint8Array)) {
          if (pos + buf.length > buffer.length) {
            if (!Buffer3.isBuffer(buf)) buf = Buffer3.from(buf);
            buf.copy(buffer, pos);
          } else {
            Uint8Array.prototype.set.call(
              buffer,
              buf,
              pos
            );
          }
        } else if (!Buffer3.isBuffer(buf)) {
          throw new TypeError('"list" argument must be an Array of Buffers');
        } else {
          buf.copy(buffer, pos);
        }
        pos += buf.length;
      }
      return buffer;
    };
    function byteLength(string, encoding) {
      if (Buffer3.isBuffer(string)) {
        return string.length;
      }
      if (ArrayBuffer.isView(string) || isInstance(string, ArrayBuffer)) {
        return string.byteLength;
      }
      if (typeof string !== "string") {
        throw new TypeError(
          'The "string" argument must be one of type string, Buffer, or ArrayBuffer. Received type ' + typeof string
        );
      }
      const len = string.length;
      const mustMatch = arguments.length > 2 && arguments[2] === true;
      if (!mustMatch && len === 0) return 0;
      let loweredCase = false;
      for (; ; ) {
        switch (encoding) {
          case "ascii":
          case "latin1":
          case "binary":
            return len;
          case "utf8":
          case "utf-8":
            return utf8ToBytes(string).length;
          case "ucs2":
          case "ucs-2":
          case "utf16le":
          case "utf-16le":
            return len * 2;
          case "hex":
            return len >>> 1;
          case "base64":
            return base64ToBytes(string).length;
          default:
            if (loweredCase) {
              return mustMatch ? -1 : utf8ToBytes(string).length;
            }
            encoding = ("" + encoding).toLowerCase();
            loweredCase = true;
        }
      }
    }
    Buffer3.byteLength = byteLength;
    function slowToString(encoding, start, end) {
      let loweredCase = false;
      if (start === void 0 || start < 0) {
        start = 0;
      }
      if (start > this.length) {
        return "";
      }
      if (end === void 0 || end > this.length) {
        end = this.length;
      }
      if (end <= 0) {
        return "";
      }
      end >>>= 0;
      start >>>= 0;
      if (end <= start) {
        return "";
      }
      if (!encoding) encoding = "utf8";
      while (true) {
        switch (encoding) {
          case "hex":
            return hexSlice(this, start, end);
          case "utf8":
          case "utf-8":
            return utf8Slice(this, start, end);
          case "ascii":
            return asciiSlice(this, start, end);
          case "latin1":
          case "binary":
            return latin1Slice(this, start, end);
          case "base64":
            return base64Slice(this, start, end);
          case "ucs2":
          case "ucs-2":
          case "utf16le":
          case "utf-16le":
            return utf16leSlice(this, start, end);
          default:
            if (loweredCase) throw new TypeError("Unknown encoding: " + encoding);
            encoding = (encoding + "").toLowerCase();
            loweredCase = true;
        }
      }
    }
    Buffer3.prototype._isBuffer = true;
    function swap(b, n, m) {
      const i = b[n];
      b[n] = b[m];
      b[m] = i;
    }
    Buffer3.prototype.swap16 = function swap16() {
      const len = this.length;
      if (len % 2 !== 0) {
        throw new RangeError("Buffer size must be a multiple of 16-bits");
      }
      for (let i = 0; i < len; i += 2) {
        swap(this, i, i + 1);
      }
      return this;
    };
    Buffer3.prototype.swap32 = function swap32() {
      const len = this.length;
      if (len % 4 !== 0) {
        throw new RangeError("Buffer size must be a multiple of 32-bits");
      }
      for (let i = 0; i < len; i += 4) {
        swap(this, i, i + 3);
        swap(this, i + 1, i + 2);
      }
      return this;
    };
    Buffer3.prototype.swap64 = function swap64() {
      const len = this.length;
      if (len % 8 !== 0) {
        throw new RangeError("Buffer size must be a multiple of 64-bits");
      }
      for (let i = 0; i < len; i += 8) {
        swap(this, i, i + 7);
        swap(this, i + 1, i + 6);
        swap(this, i + 2, i + 5);
        swap(this, i + 3, i + 4);
      }
      return this;
    };
    Buffer3.prototype.toString = function toString() {
      const length = this.length;
      if (length === 0) return "";
      if (arguments.length === 0) return utf8Slice(this, 0, length);
      return slowToString.apply(this, arguments);
    };
    Buffer3.prototype.toLocaleString = Buffer3.prototype.toString;
    Buffer3.prototype.equals = function equals(b) {
      if (!Buffer3.isBuffer(b)) throw new TypeError("Argument must be a Buffer");
      if (this === b) return true;
      return Buffer3.compare(this, b) === 0;
    };
    Buffer3.prototype.inspect = function inspect() {
      let str = "";
      const max2 = exports.INSPECT_MAX_BYTES;
      str = this.toString("hex", 0, max2).replace(/(.{2})/g, "$1 ").trim();
      if (this.length > max2) str += " ... ";
      return "<Buffer " + str + ">";
    };
    if (customInspectSymbol) {
      Buffer3.prototype[customInspectSymbol] = Buffer3.prototype.inspect;
    }
    Buffer3.prototype.compare = function compare(target, start, end, thisStart, thisEnd) {
      if (isInstance(target, Uint8Array)) {
        target = Buffer3.from(target, target.offset, target.byteLength);
      }
      if (!Buffer3.isBuffer(target)) {
        throw new TypeError(
          'The "target" argument must be one of type Buffer or Uint8Array. Received type ' + typeof target
        );
      }
      if (start === void 0) {
        start = 0;
      }
      if (end === void 0) {
        end = target ? target.length : 0;
      }
      if (thisStart === void 0) {
        thisStart = 0;
      }
      if (thisEnd === void 0) {
        thisEnd = this.length;
      }
      if (start < 0 || end > target.length || thisStart < 0 || thisEnd > this.length) {
        throw new RangeError("out of range index");
      }
      if (thisStart >= thisEnd && start >= end) {
        return 0;
      }
      if (thisStart >= thisEnd) {
        return -1;
      }
      if (start >= end) {
        return 1;
      }
      start >>>= 0;
      end >>>= 0;
      thisStart >>>= 0;
      thisEnd >>>= 0;
      if (this === target) return 0;
      let x = thisEnd - thisStart;
      let y = end - start;
      const len = Math.min(x, y);
      const thisCopy = this.slice(thisStart, thisEnd);
      const targetCopy = target.slice(start, end);
      for (let i = 0; i < len; ++i) {
        if (thisCopy[i] !== targetCopy[i]) {
          x = thisCopy[i];
          y = targetCopy[i];
          break;
        }
      }
      if (x < y) return -1;
      if (y < x) return 1;
      return 0;
    };
    function bidirectionalIndexOf(buffer, val, byteOffset, encoding, dir) {
      if (buffer.length === 0) return -1;
      if (typeof byteOffset === "string") {
        encoding = byteOffset;
        byteOffset = 0;
      } else if (byteOffset > 2147483647) {
        byteOffset = 2147483647;
      } else if (byteOffset < -2147483648) {
        byteOffset = -2147483648;
      }
      byteOffset = +byteOffset;
      if (numberIsNaN(byteOffset)) {
        byteOffset = dir ? 0 : buffer.length - 1;
      }
      if (byteOffset < 0) byteOffset = buffer.length + byteOffset;
      if (byteOffset >= buffer.length) {
        if (dir) return -1;
        else byteOffset = buffer.length - 1;
      } else if (byteOffset < 0) {
        if (dir) byteOffset = 0;
        else return -1;
      }
      if (typeof val === "string") {
        val = Buffer3.from(val, encoding);
      }
      if (Buffer3.isBuffer(val)) {
        if (val.length === 0) {
          return -1;
        }
        return arrayIndexOf(buffer, val, byteOffset, encoding, dir);
      } else if (typeof val === "number") {
        val = val & 255;
        if (typeof Uint8Array.prototype.indexOf === "function") {
          if (dir) {
            return Uint8Array.prototype.indexOf.call(buffer, val, byteOffset);
          } else {
            return Uint8Array.prototype.lastIndexOf.call(buffer, val, byteOffset);
          }
        }
        return arrayIndexOf(buffer, [val], byteOffset, encoding, dir);
      }
      throw new TypeError("val must be string, number or Buffer");
    }
    function arrayIndexOf(arr, val, byteOffset, encoding, dir) {
      let indexSize = 1;
      let arrLength = arr.length;
      let valLength = val.length;
      if (encoding !== void 0) {
        encoding = String(encoding).toLowerCase();
        if (encoding === "ucs2" || encoding === "ucs-2" || encoding === "utf16le" || encoding === "utf-16le") {
          if (arr.length < 2 || val.length < 2) {
            return -1;
          }
          indexSize = 2;
          arrLength /= 2;
          valLength /= 2;
          byteOffset /= 2;
        }
      }
      function read(buf, i2) {
        if (indexSize === 1) {
          return buf[i2];
        } else {
          return buf.readUInt16BE(i2 * indexSize);
        }
      }
      let i;
      if (dir) {
        let foundIndex = -1;
        for (i = byteOffset; i < arrLength; i++) {
          if (read(arr, i) === read(val, foundIndex === -1 ? 0 : i - foundIndex)) {
            if (foundIndex === -1) foundIndex = i;
            if (i - foundIndex + 1 === valLength) return foundIndex * indexSize;
          } else {
            if (foundIndex !== -1) i -= i - foundIndex;
            foundIndex = -1;
          }
        }
      } else {
        if (byteOffset + valLength > arrLength) byteOffset = arrLength - valLength;
        for (i = byteOffset; i >= 0; i--) {
          let found = true;
          for (let j = 0; j < valLength; j++) {
            if (read(arr, i + j) !== read(val, j)) {
              found = false;
              break;
            }
          }
          if (found) return i;
        }
      }
      return -1;
    }
    Buffer3.prototype.includes = function includes(val, byteOffset, encoding) {
      return this.indexOf(val, byteOffset, encoding) !== -1;
    };
    Buffer3.prototype.indexOf = function indexOf(val, byteOffset, encoding) {
      return bidirectionalIndexOf(this, val, byteOffset, encoding, true);
    };
    Buffer3.prototype.lastIndexOf = function lastIndexOf(val, byteOffset, encoding) {
      return bidirectionalIndexOf(this, val, byteOffset, encoding, false);
    };
    function hexWrite(buf, string, offset, length) {
      offset = Number(offset) || 0;
      const remaining = buf.length - offset;
      if (!length) {
        length = remaining;
      } else {
        length = Number(length);
        if (length > remaining) {
          length = remaining;
        }
      }
      const strLen = string.length;
      if (length > strLen / 2) {
        length = strLen / 2;
      }
      let i;
      for (i = 0; i < length; ++i) {
        const parsed = parseInt(string.substr(i * 2, 2), 16);
        if (numberIsNaN(parsed)) return i;
        buf[offset + i] = parsed;
      }
      return i;
    }
    function utf8Write(buf, string, offset, length) {
      return blitBuffer(utf8ToBytes(string, buf.length - offset), buf, offset, length);
    }
    function asciiWrite(buf, string, offset, length) {
      return blitBuffer(asciiToBytes(string), buf, offset, length);
    }
    function base64Write(buf, string, offset, length) {
      return blitBuffer(base64ToBytes(string), buf, offset, length);
    }
    function ucs2Write(buf, string, offset, length) {
      return blitBuffer(utf16leToBytes(string, buf.length - offset), buf, offset, length);
    }
    Buffer3.prototype.write = function write(string, offset, length, encoding) {
      if (offset === void 0) {
        encoding = "utf8";
        length = this.length;
        offset = 0;
      } else if (length === void 0 && typeof offset === "string") {
        encoding = offset;
        length = this.length;
        offset = 0;
      } else if (isFinite(offset)) {
        offset = offset >>> 0;
        if (isFinite(length)) {
          length = length >>> 0;
          if (encoding === void 0) encoding = "utf8";
        } else {
          encoding = length;
          length = void 0;
        }
      } else {
        throw new Error(
          "Buffer.write(string, encoding, offset[, length]) is no longer supported"
        );
      }
      const remaining = this.length - offset;
      if (length === void 0 || length > remaining) length = remaining;
      if (string.length > 0 && (length < 0 || offset < 0) || offset > this.length) {
        throw new RangeError("Attempt to write outside buffer bounds");
      }
      if (!encoding) encoding = "utf8";
      let loweredCase = false;
      for (; ; ) {
        switch (encoding) {
          case "hex":
            return hexWrite(this, string, offset, length);
          case "utf8":
          case "utf-8":
            return utf8Write(this, string, offset, length);
          case "ascii":
          case "latin1":
          case "binary":
            return asciiWrite(this, string, offset, length);
          case "base64":
            return base64Write(this, string, offset, length);
          case "ucs2":
          case "ucs-2":
          case "utf16le":
          case "utf-16le":
            return ucs2Write(this, string, offset, length);
          default:
            if (loweredCase) throw new TypeError("Unknown encoding: " + encoding);
            encoding = ("" + encoding).toLowerCase();
            loweredCase = true;
        }
      }
    };
    Buffer3.prototype.toJSON = function toJSON() {
      return {
        type: "Buffer",
        data: Array.prototype.slice.call(this._arr || this, 0)
      };
    };
    function base64Slice(buf, start, end) {
      if (start === 0 && end === buf.length) {
        return base64.fromByteArray(buf);
      } else {
        return base64.fromByteArray(buf.slice(start, end));
      }
    }
    function utf8Slice(buf, start, end) {
      end = Math.min(buf.length, end);
      const res = [];
      let i = start;
      while (i < end) {
        const firstByte = buf[i];
        let codePoint = null;
        let bytesPerSequence = firstByte > 239 ? 4 : firstByte > 223 ? 3 : firstByte > 191 ? 2 : 1;
        if (i + bytesPerSequence <= end) {
          let secondByte, thirdByte, fourthByte, tempCodePoint;
          switch (bytesPerSequence) {
            case 1:
              if (firstByte < 128) {
                codePoint = firstByte;
              }
              break;
            case 2:
              secondByte = buf[i + 1];
              if ((secondByte & 192) === 128) {
                tempCodePoint = (firstByte & 31) << 6 | secondByte & 63;
                if (tempCodePoint > 127) {
                  codePoint = tempCodePoint;
                }
              }
              break;
            case 3:
              secondByte = buf[i + 1];
              thirdByte = buf[i + 2];
              if ((secondByte & 192) === 128 && (thirdByte & 192) === 128) {
                tempCodePoint = (firstByte & 15) << 12 | (secondByte & 63) << 6 | thirdByte & 63;
                if (tempCodePoint > 2047 && (tempCodePoint < 55296 || tempCodePoint > 57343)) {
                  codePoint = tempCodePoint;
                }
              }
              break;
            case 4:
              secondByte = buf[i + 1];
              thirdByte = buf[i + 2];
              fourthByte = buf[i + 3];
              if ((secondByte & 192) === 128 && (thirdByte & 192) === 128 && (fourthByte & 192) === 128) {
                tempCodePoint = (firstByte & 15) << 18 | (secondByte & 63) << 12 | (thirdByte & 63) << 6 | fourthByte & 63;
                if (tempCodePoint > 65535 && tempCodePoint < 1114112) {
                  codePoint = tempCodePoint;
                }
              }
          }
        }
        if (codePoint === null) {
          codePoint = 65533;
          bytesPerSequence = 1;
        } else if (codePoint > 65535) {
          codePoint -= 65536;
          res.push(codePoint >>> 10 & 1023 | 55296);
          codePoint = 56320 | codePoint & 1023;
        }
        res.push(codePoint);
        i += bytesPerSequence;
      }
      return decodeCodePointsArray(res);
    }
    var MAX_ARGUMENTS_LENGTH = 4096;
    function decodeCodePointsArray(codePoints) {
      const len = codePoints.length;
      if (len <= MAX_ARGUMENTS_LENGTH) {
        return String.fromCharCode.apply(String, codePoints);
      }
      let res = "";
      let i = 0;
      while (i < len) {
        res += String.fromCharCode.apply(
          String,
          codePoints.slice(i, i += MAX_ARGUMENTS_LENGTH)
        );
      }
      return res;
    }
    function asciiSlice(buf, start, end) {
      let ret = "";
      end = Math.min(buf.length, end);
      for (let i = start; i < end; ++i) {
        ret += String.fromCharCode(buf[i] & 127);
      }
      return ret;
    }
    function latin1Slice(buf, start, end) {
      let ret = "";
      end = Math.min(buf.length, end);
      for (let i = start; i < end; ++i) {
        ret += String.fromCharCode(buf[i]);
      }
      return ret;
    }
    function hexSlice(buf, start, end) {
      const len = buf.length;
      if (!start || start < 0) start = 0;
      if (!end || end < 0 || end > len) end = len;
      let out = "";
      for (let i = start; i < end; ++i) {
        out += hexSliceLookupTable[buf[i]];
      }
      return out;
    }
    function utf16leSlice(buf, start, end) {
      const bytes = buf.slice(start, end);
      let res = "";
      for (let i = 0; i < bytes.length - 1; i += 2) {
        res += String.fromCharCode(bytes[i] + bytes[i + 1] * 256);
      }
      return res;
    }
    Buffer3.prototype.slice = function slice(start, end) {
      const len = this.length;
      start = ~~start;
      end = end === void 0 ? len : ~~end;
      if (start < 0) {
        start += len;
        if (start < 0) start = 0;
      } else if (start > len) {
        start = len;
      }
      if (end < 0) {
        end += len;
        if (end < 0) end = 0;
      } else if (end > len) {
        end = len;
      }
      if (end < start) end = start;
      const newBuf = this.subarray(start, end);
      Object.setPrototypeOf(newBuf, Buffer3.prototype);
      return newBuf;
    };
    function checkOffset(offset, ext, length) {
      if (offset % 1 !== 0 || offset < 0) throw new RangeError("offset is not uint");
      if (offset + ext > length) throw new RangeError("Trying to access beyond buffer length");
    }
    Buffer3.prototype.readUintLE = Buffer3.prototype.readUIntLE = function readUIntLE(offset, byteLength2, noAssert) {
      offset = offset >>> 0;
      byteLength2 = byteLength2 >>> 0;
      if (!noAssert) checkOffset(offset, byteLength2, this.length);
      let val = this[offset];
      let mul = 1;
      let i = 0;
      while (++i < byteLength2 && (mul *= 256)) {
        val += this[offset + i] * mul;
      }
      return val;
    };
    Buffer3.prototype.readUintBE = Buffer3.prototype.readUIntBE = function readUIntBE(offset, byteLength2, noAssert) {
      offset = offset >>> 0;
      byteLength2 = byteLength2 >>> 0;
      if (!noAssert) {
        checkOffset(offset, byteLength2, this.length);
      }
      let val = this[offset + --byteLength2];
      let mul = 1;
      while (byteLength2 > 0 && (mul *= 256)) {
        val += this[offset + --byteLength2] * mul;
      }
      return val;
    };
    Buffer3.prototype.readUint8 = Buffer3.prototype.readUInt8 = function readUInt8(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert) checkOffset(offset, 1, this.length);
      return this[offset];
    };
    Buffer3.prototype.readUint16LE = Buffer3.prototype.readUInt16LE = function readUInt16LE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert) checkOffset(offset, 2, this.length);
      return this[offset] | this[offset + 1] << 8;
    };
    Buffer3.prototype.readUint16BE = Buffer3.prototype.readUInt16BE = function readUInt16BE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert) checkOffset(offset, 2, this.length);
      return this[offset] << 8 | this[offset + 1];
    };
    Buffer3.prototype.readUint32LE = Buffer3.prototype.readUInt32LE = function readUInt32LE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert) checkOffset(offset, 4, this.length);
      return (this[offset] | this[offset + 1] << 8 | this[offset + 2] << 16) + this[offset + 3] * 16777216;
    };
    Buffer3.prototype.readUint32BE = Buffer3.prototype.readUInt32BE = function readUInt32BE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert) checkOffset(offset, 4, this.length);
      return this[offset] * 16777216 + (this[offset + 1] << 16 | this[offset + 2] << 8 | this[offset + 3]);
    };
    Buffer3.prototype.readBigUInt64LE = defineBigIntMethod(function readBigUInt64LE(offset) {
      offset = offset >>> 0;
      validateNumber(offset, "offset");
      const first = this[offset];
      const last = this[offset + 7];
      if (first === void 0 || last === void 0) {
        boundsError(offset, this.length - 8);
      }
      const lo = first + this[++offset] * 2 ** 8 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 24;
      const hi = this[++offset] + this[++offset] * 2 ** 8 + this[++offset] * 2 ** 16 + last * 2 ** 24;
      return BigInt(lo) + (BigInt(hi) << BigInt(32));
    });
    Buffer3.prototype.readBigUInt64BE = defineBigIntMethod(function readBigUInt64BE(offset) {
      offset = offset >>> 0;
      validateNumber(offset, "offset");
      const first = this[offset];
      const last = this[offset + 7];
      if (first === void 0 || last === void 0) {
        boundsError(offset, this.length - 8);
      }
      const hi = first * 2 ** 24 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + this[++offset];
      const lo = this[++offset] * 2 ** 24 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + last;
      return (BigInt(hi) << BigInt(32)) + BigInt(lo);
    });
    Buffer3.prototype.readIntLE = function readIntLE(offset, byteLength2, noAssert) {
      offset = offset >>> 0;
      byteLength2 = byteLength2 >>> 0;
      if (!noAssert) checkOffset(offset, byteLength2, this.length);
      let val = this[offset];
      let mul = 1;
      let i = 0;
      while (++i < byteLength2 && (mul *= 256)) {
        val += this[offset + i] * mul;
      }
      mul *= 128;
      if (val >= mul) val -= Math.pow(2, 8 * byteLength2);
      return val;
    };
    Buffer3.prototype.readIntBE = function readIntBE(offset, byteLength2, noAssert) {
      offset = offset >>> 0;
      byteLength2 = byteLength2 >>> 0;
      if (!noAssert) checkOffset(offset, byteLength2, this.length);
      let i = byteLength2;
      let mul = 1;
      let val = this[offset + --i];
      while (i > 0 && (mul *= 256)) {
        val += this[offset + --i] * mul;
      }
      mul *= 128;
      if (val >= mul) val -= Math.pow(2, 8 * byteLength2);
      return val;
    };
    Buffer3.prototype.readInt8 = function readInt8(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert) checkOffset(offset, 1, this.length);
      if (!(this[offset] & 128)) return this[offset];
      return (255 - this[offset] + 1) * -1;
    };
    Buffer3.prototype.readInt16LE = function readInt16LE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert) checkOffset(offset, 2, this.length);
      const val = this[offset] | this[offset + 1] << 8;
      return val & 32768 ? val | 4294901760 : val;
    };
    Buffer3.prototype.readInt16BE = function readInt16BE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert) checkOffset(offset, 2, this.length);
      const val = this[offset + 1] | this[offset] << 8;
      return val & 32768 ? val | 4294901760 : val;
    };
    Buffer3.prototype.readInt32LE = function readInt32LE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert) checkOffset(offset, 4, this.length);
      return this[offset] | this[offset + 1] << 8 | this[offset + 2] << 16 | this[offset + 3] << 24;
    };
    Buffer3.prototype.readInt32BE = function readInt32BE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert) checkOffset(offset, 4, this.length);
      return this[offset] << 24 | this[offset + 1] << 16 | this[offset + 2] << 8 | this[offset + 3];
    };
    Buffer3.prototype.readBigInt64LE = defineBigIntMethod(function readBigInt64LE(offset) {
      offset = offset >>> 0;
      validateNumber(offset, "offset");
      const first = this[offset];
      const last = this[offset + 7];
      if (first === void 0 || last === void 0) {
        boundsError(offset, this.length - 8);
      }
      const val = this[offset + 4] + this[offset + 5] * 2 ** 8 + this[offset + 6] * 2 ** 16 + (last << 24);
      return (BigInt(val) << BigInt(32)) + BigInt(first + this[++offset] * 2 ** 8 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 24);
    });
    Buffer3.prototype.readBigInt64BE = defineBigIntMethod(function readBigInt64BE(offset) {
      offset = offset >>> 0;
      validateNumber(offset, "offset");
      const first = this[offset];
      const last = this[offset + 7];
      if (first === void 0 || last === void 0) {
        boundsError(offset, this.length - 8);
      }
      const val = (first << 24) + // Overflow
      this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + this[++offset];
      return (BigInt(val) << BigInt(32)) + BigInt(this[++offset] * 2 ** 24 + this[++offset] * 2 ** 16 + this[++offset] * 2 ** 8 + last);
    });
    Buffer3.prototype.readFloatLE = function readFloatLE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert) checkOffset(offset, 4, this.length);
      return ieee754.read(this, offset, true, 23, 4);
    };
    Buffer3.prototype.readFloatBE = function readFloatBE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert) checkOffset(offset, 4, this.length);
      return ieee754.read(this, offset, false, 23, 4);
    };
    Buffer3.prototype.readDoubleLE = function readDoubleLE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert) checkOffset(offset, 8, this.length);
      return ieee754.read(this, offset, true, 52, 8);
    };
    Buffer3.prototype.readDoubleBE = function readDoubleBE(offset, noAssert) {
      offset = offset >>> 0;
      if (!noAssert) checkOffset(offset, 8, this.length);
      return ieee754.read(this, offset, false, 52, 8);
    };
    function checkInt(buf, value, offset, ext, max2, min2) {
      if (!Buffer3.isBuffer(buf)) throw new TypeError('"buffer" argument must be a Buffer instance');
      if (value > max2 || value < min2) throw new RangeError('"value" argument is out of bounds');
      if (offset + ext > buf.length) throw new RangeError("Index out of range");
    }
    Buffer3.prototype.writeUintLE = Buffer3.prototype.writeUIntLE = function writeUIntLE(value, offset, byteLength2, noAssert) {
      value = +value;
      offset = offset >>> 0;
      byteLength2 = byteLength2 >>> 0;
      if (!noAssert) {
        const maxBytes = Math.pow(2, 8 * byteLength2) - 1;
        checkInt(this, value, offset, byteLength2, maxBytes, 0);
      }
      let mul = 1;
      let i = 0;
      this[offset] = value & 255;
      while (++i < byteLength2 && (mul *= 256)) {
        this[offset + i] = value / mul & 255;
      }
      return offset + byteLength2;
    };
    Buffer3.prototype.writeUintBE = Buffer3.prototype.writeUIntBE = function writeUIntBE(value, offset, byteLength2, noAssert) {
      value = +value;
      offset = offset >>> 0;
      byteLength2 = byteLength2 >>> 0;
      if (!noAssert) {
        const maxBytes = Math.pow(2, 8 * byteLength2) - 1;
        checkInt(this, value, offset, byteLength2, maxBytes, 0);
      }
      let i = byteLength2 - 1;
      let mul = 1;
      this[offset + i] = value & 255;
      while (--i >= 0 && (mul *= 256)) {
        this[offset + i] = value / mul & 255;
      }
      return offset + byteLength2;
    };
    Buffer3.prototype.writeUint8 = Buffer3.prototype.writeUInt8 = function writeUInt8(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) checkInt(this, value, offset, 1, 255, 0);
      this[offset] = value & 255;
      return offset + 1;
    };
    Buffer3.prototype.writeUint16LE = Buffer3.prototype.writeUInt16LE = function writeUInt16LE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) checkInt(this, value, offset, 2, 65535, 0);
      this[offset] = value & 255;
      this[offset + 1] = value >>> 8;
      return offset + 2;
    };
    Buffer3.prototype.writeUint16BE = Buffer3.prototype.writeUInt16BE = function writeUInt16BE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) checkInt(this, value, offset, 2, 65535, 0);
      this[offset] = value >>> 8;
      this[offset + 1] = value & 255;
      return offset + 2;
    };
    Buffer3.prototype.writeUint32LE = Buffer3.prototype.writeUInt32LE = function writeUInt32LE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) checkInt(this, value, offset, 4, 4294967295, 0);
      this[offset + 3] = value >>> 24;
      this[offset + 2] = value >>> 16;
      this[offset + 1] = value >>> 8;
      this[offset] = value & 255;
      return offset + 4;
    };
    Buffer3.prototype.writeUint32BE = Buffer3.prototype.writeUInt32BE = function writeUInt32BE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) checkInt(this, value, offset, 4, 4294967295, 0);
      this[offset] = value >>> 24;
      this[offset + 1] = value >>> 16;
      this[offset + 2] = value >>> 8;
      this[offset + 3] = value & 255;
      return offset + 4;
    };
    function wrtBigUInt64LE(buf, value, offset, min2, max2) {
      checkIntBI(value, min2, max2, buf, offset, 7);
      let lo = Number(value & BigInt(4294967295));
      buf[offset++] = lo;
      lo = lo >> 8;
      buf[offset++] = lo;
      lo = lo >> 8;
      buf[offset++] = lo;
      lo = lo >> 8;
      buf[offset++] = lo;
      let hi = Number(value >> BigInt(32) & BigInt(4294967295));
      buf[offset++] = hi;
      hi = hi >> 8;
      buf[offset++] = hi;
      hi = hi >> 8;
      buf[offset++] = hi;
      hi = hi >> 8;
      buf[offset++] = hi;
      return offset;
    }
    function wrtBigUInt64BE(buf, value, offset, min2, max2) {
      checkIntBI(value, min2, max2, buf, offset, 7);
      let lo = Number(value & BigInt(4294967295));
      buf[offset + 7] = lo;
      lo = lo >> 8;
      buf[offset + 6] = lo;
      lo = lo >> 8;
      buf[offset + 5] = lo;
      lo = lo >> 8;
      buf[offset + 4] = lo;
      let hi = Number(value >> BigInt(32) & BigInt(4294967295));
      buf[offset + 3] = hi;
      hi = hi >> 8;
      buf[offset + 2] = hi;
      hi = hi >> 8;
      buf[offset + 1] = hi;
      hi = hi >> 8;
      buf[offset] = hi;
      return offset + 8;
    }
    Buffer3.prototype.writeBigUInt64LE = defineBigIntMethod(function writeBigUInt64LE(value, offset = 0) {
      return wrtBigUInt64LE(this, value, offset, BigInt(0), BigInt("0xffffffffffffffff"));
    });
    Buffer3.prototype.writeBigUInt64BE = defineBigIntMethod(function writeBigUInt64BE(value, offset = 0) {
      return wrtBigUInt64BE(this, value, offset, BigInt(0), BigInt("0xffffffffffffffff"));
    });
    Buffer3.prototype.writeIntLE = function writeIntLE(value, offset, byteLength2, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) {
        const limit = Math.pow(2, 8 * byteLength2 - 1);
        checkInt(this, value, offset, byteLength2, limit - 1, -limit);
      }
      let i = 0;
      let mul = 1;
      let sub = 0;
      this[offset] = value & 255;
      while (++i < byteLength2 && (mul *= 256)) {
        if (value < 0 && sub === 0 && this[offset + i - 1] !== 0) {
          sub = 1;
        }
        this[offset + i] = (value / mul >> 0) - sub & 255;
      }
      return offset + byteLength2;
    };
    Buffer3.prototype.writeIntBE = function writeIntBE(value, offset, byteLength2, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) {
        const limit = Math.pow(2, 8 * byteLength2 - 1);
        checkInt(this, value, offset, byteLength2, limit - 1, -limit);
      }
      let i = byteLength2 - 1;
      let mul = 1;
      let sub = 0;
      this[offset + i] = value & 255;
      while (--i >= 0 && (mul *= 256)) {
        if (value < 0 && sub === 0 && this[offset + i + 1] !== 0) {
          sub = 1;
        }
        this[offset + i] = (value / mul >> 0) - sub & 255;
      }
      return offset + byteLength2;
    };
    Buffer3.prototype.writeInt8 = function writeInt8(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) checkInt(this, value, offset, 1, 127, -128);
      if (value < 0) value = 255 + value + 1;
      this[offset] = value & 255;
      return offset + 1;
    };
    Buffer3.prototype.writeInt16LE = function writeInt16LE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) checkInt(this, value, offset, 2, 32767, -32768);
      this[offset] = value & 255;
      this[offset + 1] = value >>> 8;
      return offset + 2;
    };
    Buffer3.prototype.writeInt16BE = function writeInt16BE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) checkInt(this, value, offset, 2, 32767, -32768);
      this[offset] = value >>> 8;
      this[offset + 1] = value & 255;
      return offset + 2;
    };
    Buffer3.prototype.writeInt32LE = function writeInt32LE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) checkInt(this, value, offset, 4, 2147483647, -2147483648);
      this[offset] = value & 255;
      this[offset + 1] = value >>> 8;
      this[offset + 2] = value >>> 16;
      this[offset + 3] = value >>> 24;
      return offset + 4;
    };
    Buffer3.prototype.writeInt32BE = function writeInt32BE(value, offset, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) checkInt(this, value, offset, 4, 2147483647, -2147483648);
      if (value < 0) value = 4294967295 + value + 1;
      this[offset] = value >>> 24;
      this[offset + 1] = value >>> 16;
      this[offset + 2] = value >>> 8;
      this[offset + 3] = value & 255;
      return offset + 4;
    };
    Buffer3.prototype.writeBigInt64LE = defineBigIntMethod(function writeBigInt64LE(value, offset = 0) {
      return wrtBigUInt64LE(this, value, offset, -BigInt("0x8000000000000000"), BigInt("0x7fffffffffffffff"));
    });
    Buffer3.prototype.writeBigInt64BE = defineBigIntMethod(function writeBigInt64BE(value, offset = 0) {
      return wrtBigUInt64BE(this, value, offset, -BigInt("0x8000000000000000"), BigInt("0x7fffffffffffffff"));
    });
    function checkIEEE754(buf, value, offset, ext, max2, min2) {
      if (offset + ext > buf.length) throw new RangeError("Index out of range");
      if (offset < 0) throw new RangeError("Index out of range");
    }
    function writeFloat(buf, value, offset, littleEndian, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) {
        checkIEEE754(buf, value, offset, 4, 34028234663852886e22, -34028234663852886e22);
      }
      ieee754.write(buf, value, offset, littleEndian, 23, 4);
      return offset + 4;
    }
    Buffer3.prototype.writeFloatLE = function writeFloatLE(value, offset, noAssert) {
      return writeFloat(this, value, offset, true, noAssert);
    };
    Buffer3.prototype.writeFloatBE = function writeFloatBE(value, offset, noAssert) {
      return writeFloat(this, value, offset, false, noAssert);
    };
    function writeDouble(buf, value, offset, littleEndian, noAssert) {
      value = +value;
      offset = offset >>> 0;
      if (!noAssert) {
        checkIEEE754(buf, value, offset, 8, 17976931348623157e292, -17976931348623157e292);
      }
      ieee754.write(buf, value, offset, littleEndian, 52, 8);
      return offset + 8;
    }
    Buffer3.prototype.writeDoubleLE = function writeDoubleLE(value, offset, noAssert) {
      return writeDouble(this, value, offset, true, noAssert);
    };
    Buffer3.prototype.writeDoubleBE = function writeDoubleBE(value, offset, noAssert) {
      return writeDouble(this, value, offset, false, noAssert);
    };
    Buffer3.prototype.copy = function copy(target, targetStart, start, end) {
      if (!Buffer3.isBuffer(target)) throw new TypeError("argument should be a Buffer");
      if (!start) start = 0;
      if (!end && end !== 0) end = this.length;
      if (targetStart >= target.length) targetStart = target.length;
      if (!targetStart) targetStart = 0;
      if (end > 0 && end < start) end = start;
      if (end === start) return 0;
      if (target.length === 0 || this.length === 0) return 0;
      if (targetStart < 0) {
        throw new RangeError("targetStart out of bounds");
      }
      if (start < 0 || start >= this.length) throw new RangeError("Index out of range");
      if (end < 0) throw new RangeError("sourceEnd out of bounds");
      if (end > this.length) end = this.length;
      if (target.length - targetStart < end - start) {
        end = target.length - targetStart + start;
      }
      const len = end - start;
      if (this === target && typeof Uint8Array.prototype.copyWithin === "function") {
        this.copyWithin(targetStart, start, end);
      } else {
        Uint8Array.prototype.set.call(
          target,
          this.subarray(start, end),
          targetStart
        );
      }
      return len;
    };
    Buffer3.prototype.fill = function fill(val, start, end, encoding) {
      if (typeof val === "string") {
        if (typeof start === "string") {
          encoding = start;
          start = 0;
          end = this.length;
        } else if (typeof end === "string") {
          encoding = end;
          end = this.length;
        }
        if (encoding !== void 0 && typeof encoding !== "string") {
          throw new TypeError("encoding must be a string");
        }
        if (typeof encoding === "string" && !Buffer3.isEncoding(encoding)) {
          throw new TypeError("Unknown encoding: " + encoding);
        }
        if (val.length === 1) {
          const code = val.charCodeAt(0);
          if (encoding === "utf8" && code < 128 || encoding === "latin1") {
            val = code;
          }
        }
      } else if (typeof val === "number") {
        val = val & 255;
      } else if (typeof val === "boolean") {
        val = Number(val);
      }
      if (start < 0 || this.length < start || this.length < end) {
        throw new RangeError("Out of range index");
      }
      if (end <= start) {
        return this;
      }
      start = start >>> 0;
      end = end === void 0 ? this.length : end >>> 0;
      if (!val) val = 0;
      let i;
      if (typeof val === "number") {
        for (i = start; i < end; ++i) {
          this[i] = val;
        }
      } else {
        const bytes = Buffer3.isBuffer(val) ? val : Buffer3.from(val, encoding);
        const len = bytes.length;
        if (len === 0) {
          throw new TypeError('The value "' + val + '" is invalid for argument "value"');
        }
        for (i = 0; i < end - start; ++i) {
          this[i + start] = bytes[i % len];
        }
      }
      return this;
    };
    var errors = {};
    function E(sym, getMessage, Base) {
      errors[sym] = class NodeError extends Base {
        constructor() {
          super();
          Object.defineProperty(this, "message", {
            value: getMessage.apply(this, arguments),
            writable: true,
            configurable: true
          });
          this.name = `${this.name} [${sym}]`;
          this.stack;
          delete this.name;
        }
        get code() {
          return sym;
        }
        set code(value) {
          Object.defineProperty(this, "code", {
            configurable: true,
            enumerable: true,
            value,
            writable: true
          });
        }
        toString() {
          return `${this.name} [${sym}]: ${this.message}`;
        }
      };
    }
    E(
      "ERR_BUFFER_OUT_OF_BOUNDS",
      function(name) {
        if (name) {
          return `${name} is outside of buffer bounds`;
        }
        return "Attempt to access memory outside buffer bounds";
      },
      RangeError
    );
    E(
      "ERR_INVALID_ARG_TYPE",
      function(name, actual) {
        return `The "${name}" argument must be of type number. Received type ${typeof actual}`;
      },
      TypeError
    );
    E(
      "ERR_OUT_OF_RANGE",
      function(str, range, input) {
        let msg = `The value of "${str}" is out of range.`;
        let received = input;
        if (Number.isInteger(input) && Math.abs(input) > 2 ** 32) {
          received = addNumericalSeparator(String(input));
        } else if (typeof input === "bigint") {
          received = String(input);
          if (input > BigInt(2) ** BigInt(32) || input < -(BigInt(2) ** BigInt(32))) {
            received = addNumericalSeparator(received);
          }
          received += "n";
        }
        msg += ` It must be ${range}. Received ${received}`;
        return msg;
      },
      RangeError
    );
    function addNumericalSeparator(val) {
      let res = "";
      let i = val.length;
      const start = val[0] === "-" ? 1 : 0;
      for (; i >= start + 4; i -= 3) {
        res = `_${val.slice(i - 3, i)}${res}`;
      }
      return `${val.slice(0, i)}${res}`;
    }
    function checkBounds(buf, offset, byteLength2) {
      validateNumber(offset, "offset");
      if (buf[offset] === void 0 || buf[offset + byteLength2] === void 0) {
        boundsError(offset, buf.length - (byteLength2 + 1));
      }
    }
    function checkIntBI(value, min2, max2, buf, offset, byteLength2) {
      if (value > max2 || value < min2) {
        const n = typeof min2 === "bigint" ? "n" : "";
        let range;
        if (byteLength2 > 3) {
          if (min2 === 0 || min2 === BigInt(0)) {
            range = `>= 0${n} and < 2${n} ** ${(byteLength2 + 1) * 8}${n}`;
          } else {
            range = `>= -(2${n} ** ${(byteLength2 + 1) * 8 - 1}${n}) and < 2 ** ${(byteLength2 + 1) * 8 - 1}${n}`;
          }
        } else {
          range = `>= ${min2}${n} and <= ${max2}${n}`;
        }
        throw new errors.ERR_OUT_OF_RANGE("value", range, value);
      }
      checkBounds(buf, offset, byteLength2);
    }
    function validateNumber(value, name) {
      if (typeof value !== "number") {
        throw new errors.ERR_INVALID_ARG_TYPE(name, "number", value);
      }
    }
    function boundsError(value, length, type) {
      if (Math.floor(value) !== value) {
        validateNumber(value, type);
        throw new errors.ERR_OUT_OF_RANGE(type || "offset", "an integer", value);
      }
      if (length < 0) {
        throw new errors.ERR_BUFFER_OUT_OF_BOUNDS();
      }
      throw new errors.ERR_OUT_OF_RANGE(
        type || "offset",
        `>= ${type ? 1 : 0} and <= ${length}`,
        value
      );
    }
    var INVALID_BASE64_RE = /[^+/0-9A-Za-z-_]/g;
    function base64clean(str) {
      str = str.split("=")[0];
      str = str.trim().replace(INVALID_BASE64_RE, "");
      if (str.length < 2) return "";
      while (str.length % 4 !== 0) {
        str = str + "=";
      }
      return str;
    }
    function utf8ToBytes(string, units) {
      units = units || Infinity;
      let codePoint;
      const length = string.length;
      let leadSurrogate = null;
      const bytes = [];
      for (let i = 0; i < length; ++i) {
        codePoint = string.charCodeAt(i);
        if (codePoint > 55295 && codePoint < 57344) {
          if (!leadSurrogate) {
            if (codePoint > 56319) {
              if ((units -= 3) > -1) bytes.push(239, 191, 189);
              continue;
            } else if (i + 1 === length) {
              if ((units -= 3) > -1) bytes.push(239, 191, 189);
              continue;
            }
            leadSurrogate = codePoint;
            continue;
          }
          if (codePoint < 56320) {
            if ((units -= 3) > -1) bytes.push(239, 191, 189);
            leadSurrogate = codePoint;
            continue;
          }
          codePoint = (leadSurrogate - 55296 << 10 | codePoint - 56320) + 65536;
        } else if (leadSurrogate) {
          if ((units -= 3) > -1) bytes.push(239, 191, 189);
        }
        leadSurrogate = null;
        if (codePoint < 128) {
          if ((units -= 1) < 0) break;
          bytes.push(codePoint);
        } else if (codePoint < 2048) {
          if ((units -= 2) < 0) break;
          bytes.push(
            codePoint >> 6 | 192,
            codePoint & 63 | 128
          );
        } else if (codePoint < 65536) {
          if ((units -= 3) < 0) break;
          bytes.push(
            codePoint >> 12 | 224,
            codePoint >> 6 & 63 | 128,
            codePoint & 63 | 128
          );
        } else if (codePoint < 1114112) {
          if ((units -= 4) < 0) break;
          bytes.push(
            codePoint >> 18 | 240,
            codePoint >> 12 & 63 | 128,
            codePoint >> 6 & 63 | 128,
            codePoint & 63 | 128
          );
        } else {
          throw new Error("Invalid code point");
        }
      }
      return bytes;
    }
    function asciiToBytes(str) {
      const byteArray = [];
      for (let i = 0; i < str.length; ++i) {
        byteArray.push(str.charCodeAt(i) & 255);
      }
      return byteArray;
    }
    function utf16leToBytes(str, units) {
      let c, hi, lo;
      const byteArray = [];
      for (let i = 0; i < str.length; ++i) {
        if ((units -= 2) < 0) break;
        c = str.charCodeAt(i);
        hi = c >> 8;
        lo = c % 256;
        byteArray.push(lo);
        byteArray.push(hi);
      }
      return byteArray;
    }
    function base64ToBytes(str) {
      return base64.toByteArray(base64clean(str));
    }
    function blitBuffer(src, dst, offset, length) {
      let i;
      for (i = 0; i < length; ++i) {
        if (i + offset >= dst.length || i >= src.length) break;
        dst[i + offset] = src[i];
      }
      return i;
    }
    function isInstance(obj, type) {
      return obj instanceof type || obj != null && obj.constructor != null && obj.constructor.name != null && obj.constructor.name === type.name;
    }
    function numberIsNaN(obj) {
      return obj !== obj;
    }
    var hexSliceLookupTable = (function() {
      const alphabet = "0123456789abcdef";
      const table = new Array(256);
      for (let i = 0; i < 16; ++i) {
        const i16 = i * 16;
        for (let j = 0; j < 16; ++j) {
          table[i16 + j] = alphabet[i] + alphabet[j];
        }
      }
      return table;
    })();
    function defineBigIntMethod(fn) {
      return typeof BigInt === "undefined" ? BufferBigIntNotDefined : fn;
    }
    function BufferBigIntNotDefined() {
      throw new Error("BigInt not supported");
    }
  }
});

// scripts/buffer-shim.js
var import_buffer;
var init_buffer_shim = __esm({
  "scripts/buffer-shim.js"() {
    import_buffer = __toESM(require_buffer(), 1);
  }
});

// src/parser/ast.js
function ExplainStmt(query) {
  return { kind: NodeKind.EXPLAIN_STMT, query };
}
function SelectStmt({
  withClause = null,
  distinct = false,
  selectItems,
  from = null,
  where = null,
  groupBy = null,
  having = null,
  orderBy = null,
  limit = null,
  offset = null
}) {
  return { kind: NodeKind.SELECT_STMT, withClause, distinct, selectItems, from, where, groupBy, having, orderBy, limit, offset };
}
function SetOp(op, left, right, all = false) {
  return { kind: NodeKind.SET_OP, op, left, right, all };
}
function WithClause(ctes) {
  return { kind: NodeKind.WITH_CLAUSE, ctes };
}
function CTE(name, query, columnAliases = null) {
  return { kind: NodeKind.CTE, name, query, columnAliases };
}
function SelectItem(expr2, alias = null) {
  return { kind: NodeKind.SELECT_ITEM, expr: expr2, alias };
}
function AllColumns(table = null) {
  return { kind: NodeKind.ALL_COLUMNS, table };
}
function TableRef(name, alias = null) {
  return { kind: NodeKind.TABLE_REF, name, alias: alias || name };
}
function JoinRef(left, right, joinType, condition = null) {
  return { kind: NodeKind.JOIN_REF, left, right, joinType, condition };
}
function SubqueryRef(query, alias) {
  return { kind: NodeKind.SUBQUERY_REF, query, alias };
}
function ColumnRef(name, table = null) {
  return { kind: NodeKind.COLUMN_REF, name, table };
}
function Literal(value, dataType = null) {
  return { kind: NodeKind.LITERAL, value, dataType };
}
function BinaryExpr(op, left, right) {
  return { kind: NodeKind.BINARY_EXPR, op, left, right };
}
function UnaryExpr(op, operand) {
  return { kind: NodeKind.UNARY_EXPR, op, operand };
}
function BetweenExpr(expr2, low, high, negated = false) {
  return { kind: NodeKind.BETWEEN_EXPR, expr: expr2, low, high, negated };
}
function InExpr(expr2, list, negated = false) {
  return { kind: NodeKind.IN_EXPR, expr: expr2, list, negated };
}
function LikeExpr(expr2, pattern, negated = false) {
  return { kind: NodeKind.LIKE_EXPR, expr: expr2, pattern, negated };
}
function IsNullExpr(expr2, negated = false) {
  return { kind: NodeKind.IS_NULL_EXPR, expr: expr2, negated };
}
function ExistsExpr(query, negated = false) {
  return { kind: NodeKind.EXISTS_EXPR, query, negated };
}
function SubqueryExpr(query) {
  return { kind: NodeKind.SUBQUERY_EXPR, query };
}
function FunctionCall(name, args) {
  return { kind: NodeKind.FUNCTION_CALL, name, args };
}
function AggregateCall(name, args, distinct = false) {
  return { kind: NodeKind.AGGREGATE_CALL, name, args, distinct };
}
function CaseExpr(operand, whenClauses, elseExpr = null) {
  return { kind: NodeKind.CASE_EXPR, operand, whenClauses, elseExpr };
}
function WhenClause(condition, result) {
  return { kind: NodeKind.WHEN_CLAUSE, condition, result };
}
function CastExpr(expr2, targetType) {
  return { kind: NodeKind.CAST_EXPR, expr: expr2, targetType };
}
function ExtractExpr(field, source) {
  return { kind: NodeKind.EXTRACT_EXPR, field, source };
}
function SubstringExpr(expr2, from, length = null) {
  return { kind: NodeKind.SUBSTRING_EXPR, expr: expr2, from, length };
}
function IntervalExpr(value, unit) {
  return { kind: NodeKind.INTERVAL_EXPR, value, unit };
}
function OrderKey(expr2, direction = "ASC", nullOrder = null) {
  return { kind: NodeKind.ORDER_KEY, expr: expr2, direction, nullOrder };
}
function TypeName(name, params = []) {
  return { kind: NodeKind.TYPE_NAME, name, params };
}
function WindowSpec(partitionBy = [], orderBy = []) {
  return { kind: NodeKind.WINDOW_SPEC, partitionBy, orderBy };
}
function WindowCall(name, args, windowSpec) {
  return { kind: NodeKind.WINDOW_CALL, name, args, windowSpec };
}
function CreateTableStmt(name, columns, ifNotExists = false) {
  return { kind: NodeKind.CREATE_TABLE_STMT, name, columns, ifNotExists };
}
function DropTableStmt(name, ifExists = false) {
  return { kind: NodeKind.DROP_TABLE_STMT, name, ifExists };
}
function ColumnDef(name, typeName) {
  return { kind: NodeKind.COLUMN_DEF, name, typeName };
}
function ExplainAnalyzeStmt(query) {
  return { kind: NodeKind.EXPLAIN_ANALYZE_STMT, query };
}
var NodeKind;
var init_ast = __esm({
  "src/parser/ast.js"() {
    init_buffer_shim();
    NodeKind = {
      EXPLAIN_STMT: "ExplainStmt",
      SELECT_STMT: "SelectStmt",
      SET_OP: "SetOp",
      WITH_CLAUSE: "WithClause",
      CTE: "CTE",
      SELECT_ITEM: "SelectItem",
      ALL_COLUMNS: "AllColumns",
      TABLE_REF: "TableRef",
      JOIN_REF: "JoinRef",
      SUBQUERY_REF: "SubqueryRef",
      COLUMN_REF: "ColumnRef",
      LITERAL: "Literal",
      BINARY_EXPR: "BinaryExpr",
      UNARY_EXPR: "UnaryExpr",
      BETWEEN_EXPR: "BetweenExpr",
      IN_EXPR: "InExpr",
      LIKE_EXPR: "LikeExpr",
      IS_NULL_EXPR: "IsNullExpr",
      EXISTS_EXPR: "ExistsExpr",
      SUBQUERY_EXPR: "SubqueryExpr",
      FUNCTION_CALL: "FunctionCall",
      AGGREGATE_CALL: "AggregateCall",
      CASE_EXPR: "CaseExpr",
      WHEN_CLAUSE: "WhenClause",
      CAST_EXPR: "CastExpr",
      EXTRACT_EXPR: "ExtractExpr",
      SUBSTRING_EXPR: "SubstringExpr",
      INTERVAL_EXPR: "IntervalExpr",
      ORDER_KEY: "OrderKey",
      TYPE_NAME: "TypeName",
      WINDOW_SPEC: "WindowSpec",
      WINDOW_CALL: "WindowCall",
      CREATE_TABLE_STMT: "CreateTableStmt",
      DROP_TABLE_STMT: "DropTableStmt",
      COLUMN_DEF: "ColumnDef",
      EXPLAIN_ANALYZE_STMT: "ExplainAnalyzeStmt"
    };
  }
});

// src/storage/data-type.js
function typedArrayCtorFor(dataType) {
  const Ctor = TYPE_TO_ARRAY[dataType];
  if (!Ctor) {
    throw new Error(`No TypedArray for type ${dataType}`);
  }
  return Ctor;
}
function byteWidthFor(dataType) {
  return TYPE_TO_BYTE_WIDTH[dataType] ?? 0;
}
function isFixedWidth(dataType) {
  return FIXED_WIDTH_TYPES.has(dataType);
}
function isNumeric(dataType) {
  return dataType === DataType.INT32 || dataType === DataType.INT64 || dataType === DataType.FLOAT64 || dataType === DataType.DECIMAL;
}
function isTemporal(dataType) {
  return dataType === DataType.DATE || dataType === DataType.TIMESTAMP;
}
function isComparable(a, b) {
  if (a === b) return true;
  if (isNumeric(a) && isNumeric(b)) return true;
  if (isTemporal(a) && isTemporal(b)) return true;
  return false;
}
function dateToEpochDays(year, month, day) {
  const ms = Date.UTC(year, month - 1, day);
  return Math.floor(ms / 864e5);
}
function epochDaysToDate(days) {
  const ms = days * 864e5;
  const d = new Date(ms);
  return {
    year: d.getUTCFullYear(),
    month: d.getUTCMonth() + 1,
    day: d.getUTCDate()
  };
}
function timestampToEpochMs(year, month, day, hour, minute, second, ms = 0) {
  return Date.UTC(year, month - 1, day, hour, minute, second, ms);
}
function epochMsToTimestamp(epochMs) {
  const d = new Date(epochMs);
  return {
    year: d.getUTCFullYear(),
    month: d.getUTCMonth() + 1,
    day: d.getUTCDate(),
    hour: d.getUTCHours(),
    minute: d.getUTCMinutes(),
    second: d.getUTCSeconds(),
    ms: d.getUTCMilliseconds()
  };
}
var DataType, FIXED_WIDTH_TYPES, TYPE_TO_ARRAY, TYPE_TO_BYTE_WIDTH, DECIMAL_SCALE_NUMBER;
var init_data_type = __esm({
  "src/storage/data-type.js"() {
    init_buffer_shim();
    DataType = {
      BOOLEAN: "BOOLEAN",
      INT32: "INT32",
      INT64: "INT64",
      FLOAT64: "FLOAT64",
      DECIMAL: "DECIMAL",
      VARCHAR: "VARCHAR",
      DATE: "DATE",
      TIMESTAMP: "TIMESTAMP"
    };
    FIXED_WIDTH_TYPES = /* @__PURE__ */ new Set([
      DataType.BOOLEAN,
      DataType.INT32,
      DataType.INT64,
      DataType.FLOAT64,
      DataType.DECIMAL,
      DataType.DATE,
      DataType.TIMESTAMP
    ]);
    TYPE_TO_ARRAY = {
      [DataType.BOOLEAN]: Uint8Array,
      [DataType.INT32]: Int32Array,
      [DataType.INT64]: BigInt64Array,
      [DataType.FLOAT64]: Float64Array,
      [DataType.DECIMAL]: BigInt64Array,
      [DataType.DATE]: Int32Array,
      [DataType.TIMESTAMP]: BigInt64Array
    };
    TYPE_TO_BYTE_WIDTH = {
      [DataType.BOOLEAN]: 1,
      [DataType.INT32]: 4,
      [DataType.INT64]: 8,
      [DataType.FLOAT64]: 8,
      [DataType.DECIMAL]: 8,
      [DataType.DATE]: 4,
      [DataType.TIMESTAMP]: 8
    };
    DECIMAL_SCALE_NUMBER = 100;
  }
});

// src/binder/expression-binder.js
function BoundColumnRef(tableAlias, columnName, columnIndex, dataType, depth = 0) {
  return {
    kind: BoundExprKind.COLUMN_REF,
    tableAlias,
    columnName,
    columnIndex,
    dataType,
    depth,
    isCorrelated: depth > 0
  };
}
function BoundLiteral(value, dataType) {
  return { kind: BoundExprKind.LITERAL, value, dataType };
}
function BoundBinary(op, left, right, resultType) {
  return { kind: BoundExprKind.BINARY, op, left, right, resultType };
}
function BoundUnary(op, operand, resultType) {
  return { kind: BoundExprKind.UNARY, op, operand, resultType };
}
function BoundFunction(name, args, resultType) {
  return { kind: BoundExprKind.FUNCTION, name, args, resultType };
}
function BoundAggregate(name, args, distinct, resultType) {
  return { kind: BoundExprKind.AGGREGATE, name, args, distinct, resultType };
}
function BoundCase(operand, whenClauses, elseExpr, resultType) {
  return { kind: BoundExprKind.CASE, operand, whenClauses, elseExpr, resultType };
}
function BoundCast(expr2, targetType) {
  return { kind: BoundExprKind.CAST, expr: expr2, targetType, dataType: targetType };
}
function BoundBetween(expr2, low, high, negated) {
  return { kind: BoundExprKind.BETWEEN, expr: expr2, low, high, negated, resultType: DataType.BOOLEAN };
}
function BoundInList(expr2, list, negated) {
  return { kind: BoundExprKind.IN_LIST, expr: expr2, list, negated, resultType: DataType.BOOLEAN };
}
function BoundLike(expr2, pattern, negated) {
  return { kind: BoundExprKind.LIKE, expr: expr2, pattern, negated, resultType: DataType.BOOLEAN };
}
function BoundIsNull(expr2, negated) {
  return { kind: BoundExprKind.IS_NULL, expr: expr2, negated, resultType: DataType.BOOLEAN };
}
function BoundSubquery(plan, subqueryType) {
  return { kind: BoundExprKind.SUBQUERY, plan, subqueryType };
}
function BoundExists(plan, negated) {
  return { kind: BoundExprKind.EXISTS, plan, negated, resultType: DataType.BOOLEAN };
}
function BoundExtract(field, source) {
  return { kind: BoundExprKind.EXTRACT, field, source, resultType: DataType.INT32 };
}
function BoundInterval(value, unit) {
  return { kind: BoundExprKind.INTERVAL, value, unit, resultType: DataType.INT32 };
}
function BoundWindow(name, args, partitionBy, orderBy, resultType) {
  return { kind: BoundExprKind.WINDOW, name, args, partitionBy, orderBy, resultType };
}
function getExprType(expr2) {
  if (!expr2) return null;
  return expr2.resultType || expr2.dataType || null;
}
var BoundExprKind;
var init_expression_binder = __esm({
  "src/binder/expression-binder.js"() {
    init_buffer_shim();
    init_ast();
    init_data_type();
    BoundExprKind = {
      COLUMN_REF: "BoundColumnRef",
      LITERAL: "BoundLiteral",
      BINARY: "BoundBinary",
      UNARY: "BoundUnary",
      FUNCTION: "BoundFunction",
      AGGREGATE: "BoundAggregate",
      CASE: "BoundCase",
      CAST: "BoundCast",
      BETWEEN: "BoundBetween",
      IN_LIST: "BoundInList",
      LIKE: "BoundLike",
      IS_NULL: "BoundIsNull",
      SUBQUERY: "BoundSubquery",
      EXISTS: "BoundExists",
      EXTRACT: "BoundExtract",
      INTERVAL: "BoundInterval",
      COMPARISON: "BoundComparison",
      WINDOW: "BoundWindow"
    };
  }
});

// src/planner/logical-plan.js
function LogicalScan(table, columns, alias) {
  return { type: PlanNodeType.SCAN, table, columns, alias: alias || table };
}
function LogicalFilter(condition, child) {
  return { type: PlanNodeType.FILTER, condition, children: [child] };
}
function LogicalProject(expressions, child) {
  return { type: PlanNodeType.PROJECT, expressions, children: [child] };
}
function LogicalJoin(joinType, condition, left, right, physicalStrategy = PhysicalStrategy.HASH) {
  return {
    type: PlanNodeType.JOIN,
    joinType,
    condition,
    children: [left, right],
    physicalStrategy
  };
}
function LogicalAggregate(groupBy, aggregates, child, physicalStrategy = PhysicalStrategy.HASH) {
  return {
    type: PlanNodeType.AGGREGATE,
    groupBy,
    aggregates,
    children: [child],
    physicalStrategy
  };
}
function LogicalSort(orderKeys, child) {
  return { type: PlanNodeType.SORT, orderKeys, children: [child] };
}
function LogicalLimit(count2, offset, child) {
  return { type: PlanNodeType.LIMIT, count: count2, offset: offset || 0, children: [child] };
}
function LogicalDistinct(child) {
  return { type: PlanNodeType.DISTINCT, children: [child] };
}
function LogicalUnion(left, right, all) {
  return { type: PlanNodeType.UNION, all: !!all, children: [left, right] };
}
function LogicalCTEScan(cteName, cteId) {
  return { type: PlanNodeType.CTE_SCAN, cteName, cteId, children: [] };
}
function LogicalDependentJoin(child, subquery, correlatedColumns, subqueryType, condition) {
  return {
    type: PlanNodeType.DEPENDENT_JOIN,
    correlatedColumns,
    subqueryType,
    condition,
    children: [child, subquery]
  };
}
function LogicalIndexScan(table, alias, indexName, columnName, scanType, scanKey, scanLow, scanHigh, lowInc, highInc, columns) {
  return {
    type: PlanNodeType.INDEX_SCAN,
    table,
    alias: alias || table,
    indexName,
    columnName,
    scanType,
    scanKey,
    scanLow,
    scanHigh,
    lowInc,
    highInc,
    columns
  };
}
function LogicalWindow(windowExprs, child) {
  return { type: PlanNodeType.WINDOW, windowExprs, children: [child] };
}
function LogicalMaterialize(child) {
  return { type: PlanNodeType.MATERIALIZE, children: [child] };
}
function LogicalSingleRow() {
  return { type: PlanNodeType.SINGLE_ROW, children: [] };
}
function getChildren(node) {
  return node.children || [];
}
function setChildren(node, children) {
  return { ...node, children };
}
function planToString(node, indent = 0) {
  const prefix = "  ".repeat(indent);
  let str = `${prefix}${node.type}`;
  switch (node.type) {
    case PlanNodeType.SCAN:
      str += `(${node.table}${node.alias !== node.table ? ` AS ${node.alias}` : ""})`;
      break;
    case PlanNodeType.JOIN:
      str += `(${node.joinType})`;
      break;
    case PlanNodeType.LIMIT:
      str += `(${node.count}${node.offset ? `, offset=${node.offset}` : ""})`;
      break;
    case PlanNodeType.CTE_SCAN:
      str += `(${node.cteName})`;
      break;
    case PlanNodeType.CTE_ANCHOR:
      str += `(${node.cteName})`;
      break;
    case PlanNodeType.DEPENDENT_JOIN:
      str += `(${node.subqueryType})`;
      break;
    case PlanNodeType.WINDOW:
      str += `(${node.windowExprs.map((w) => w.name).join(", ")})`;
      break;
    case PlanNodeType.TOP_N:
      str += `(${node.count}${node.offset ? `, offset=${node.offset}` : ""})`;
      break;
    case PlanNodeType.INDEX_SCAN:
      str += `(${node.table} using ${node.indexName})`;
      break;
    case PlanNodeType.EXCHANGE:
      str += `(${node.exchangeType})`;
      break;
    case PlanNodeType.PARTIAL_AGGREGATE:
      str += `(partial)`;
      break;
    case PlanNodeType.FINAL_AGGREGATE:
      str += `(final)`;
      break;
    case PlanNodeType.MERGE_EXCHANGE:
      str += `(${node.limit ? `limit=${node.limit}` : "all"})`;
      break;
    case PlanNodeType.EXCHANGE_RECEIVE:
      str += `(fragments=[${node.sourceFragmentIds.join(",")}])`;
      break;
  }
  str += "\n";
  for (const child of getChildren(node)) {
    str += planToString(child, indent + 1);
  }
  return str;
}
var PlanNodeType, JoinType, PhysicalStrategy, SortDirection;
var init_logical_plan = __esm({
  "src/planner/logical-plan.js"() {
    init_buffer_shim();
    PlanNodeType = {
      SCAN: "Scan",
      FILTER: "Filter",
      PROJECT: "Project",
      JOIN: "Join",
      AGGREGATE: "Aggregate",
      SORT: "Sort",
      LIMIT: "Limit",
      DISTINCT: "Distinct",
      UNION: "Union",
      CTE_SCAN: "CTEScan",
      CTE_ANCHOR: "CTEAnchor",
      DEPENDENT_JOIN: "DependentJoin",
      MATERIALIZE: "Materialize",
      EMPTY: "Empty",
      TOP_N: "TopN",
      INDEX_SCAN: "IndexScan",
      WINDOW: "Window",
      EXCHANGE: "Exchange",
      PARTIAL_AGGREGATE: "PartialAggregate",
      FINAL_AGGREGATE: "FinalAggregate",
      MERGE_EXCHANGE: "MergeExchange",
      EXCHANGE_RECEIVE: "ExchangeReceive",
      SINGLE_ROW: "SingleRow"
    };
    JoinType = {
      INNER: "INNER",
      LEFT: "LEFT",
      RIGHT: "RIGHT",
      FULL: "FULL",
      SEMI: "SEMI",
      ANTI: "ANTI",
      MARK: "MARK",
      SINGLE: "SINGLE",
      CROSS: "CROSS"
    };
    PhysicalStrategy = {
      HASH: "HASH",
      MERGE: "MERGE",
      NESTED_LOOP: "NESTED_LOOP",
      STREAM: "STREAM",
      UNGROUPED: "UNGROUPED",
      PERFECT_HASH: "PERFECT_HASH"
    };
    SortDirection = {
      ASC: "ASC",
      DESC: "DESC"
    };
  }
});

// src/utils/bitmap.js
function bitmapWordCount(length) {
  return Math.ceil(length / 32);
}
function setBit(bitmap, index) {
  bitmap[index >>> 5] |= 1 << (index & 31);
}
function clearBit(bitmap, index) {
  bitmap[index >>> 5] &= ~(1 << (index & 31));
}
function testBit(bitmap, index) {
  return (bitmap[index >>> 5] & 1 << (index & 31)) !== 0;
}
var POPCOUNT_TABLE;
var init_bitmap = __esm({
  "src/utils/bitmap.js"() {
    init_buffer_shim();
    POPCOUNT_TABLE = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
      POPCOUNT_TABLE[i] = POPCOUNT_TABLE[i >> 1] + (i & 1);
    }
  }
});

// src/storage/dictionary-column.js
var DEFAULT_CAPACITY, MAX_DICT_SIZE, DictionaryColumn;
var init_dictionary_column = __esm({
  "src/storage/dictionary-column.js"() {
    init_buffer_shim();
    init_data_type();
    init_bitmap();
    init_sab_arena();
    DEFAULT_CAPACITY = 2048;
    MAX_DICT_SIZE = 65535;
    DictionaryColumn = class _DictionaryColumn {
      constructor(capacity = DEFAULT_CAPACITY, allocator = heapAllocator) {
        this.dataType = DataType.VARCHAR;
        this.capacity = capacity;
        this.length = 0;
        this.allocator = allocator;
        this._dictionary = /* @__PURE__ */ new Map();
        this.reverseDict = [];
        this.indices = allocator.acquire(Uint16Array, capacity);
        this.nullBitmap = allocator.acquire(Uint32Array, bitmapWordCount(capacity));
        this.hasNulls = false;
      }
      static fromParts({ indices, reverseDict, nullBitmap, length, hasNulls, allocator }) {
        const col2 = Object.create(_DictionaryColumn.prototype);
        col2.dataType = DataType.VARCHAR;
        col2.capacity = length;
        col2.length = length;
        col2.allocator = allocator || heapAllocator;
        col2._dictionary = null;
        col2.reverseDict = reverseDict;
        col2.indices = indices;
        col2.nullBitmap = nullBitmap;
        col2.hasNulls = hasNulls;
        return col2;
      }
      get dictionary() {
        if (this._dictionary === null) {
          this._dictionary = new Map(this.reverseDict.map((value, id) => [value, id]));
        }
        return this._dictionary;
      }
      set dictionary(map) {
        this._dictionary = map;
      }
      get(index) {
        if (this.hasNulls && !testBit(this.nullBitmap, index)) {
          return null;
        }
        const dictId = this.indices[index];
        return this.reverseDict[dictId];
      }
      set(index, value) {
        if (value === null || value === void 0) {
          this._setNull(index);
          return;
        }
        setBit(this.nullBitmap, index);
        let dictId = this.dictionary.get(value);
        if (dictId === void 0) {
          dictId = this.reverseDict.length;
          if (dictId > MAX_DICT_SIZE) {
            throw new Error(`Dictionary capacity exceeded ${MAX_DICT_SIZE} values per chunk`);
          }
          this.dictionary.set(value, dictId);
          this.reverseDict.push(value);
        }
        this.indices[index] = dictId;
        if (index >= this.length) {
          this.length = index + 1;
        }
      }
      append(value) {
        if (this.length >= this.capacity) {
          this._grow();
        }
        this.set(this.length, value);
      }
      appendBatch(values) {
        for (let i = 0; i < values.length; i++) {
          this.append(values[i]);
        }
      }
      isNull(index) {
        return this.hasNulls && !testBit(this.nullBitmap, index);
      }
      slice(start, end) {
        const len = end - start;
        const col2 = new _DictionaryColumn(len, this.allocator);
        col2.dictionary = this.dictionary;
        col2.reverseDict = this.reverseDict;
        const srcSlice = this.indices.subarray(start, end);
        col2.indices.set(srcSlice);
        for (let i = 0; i < len; i++) {
          if (testBit(this.nullBitmap, start + i)) {
            setBit(col2.nullBitmap, i);
          }
        }
        col2.length = len;
        col2.hasNulls = this.hasNulls;
        return col2;
      }
      _setNull(index) {
        this.hasNulls = true;
        clearBit(this.nullBitmap, index);
        if (index >= this.length) {
          this.length = index + 1;
        }
      }
      _grow() {
        const newCapacity = this.capacity * 2;
        const newIndices = this.allocator.acquire(Uint16Array, newCapacity);
        newIndices.set(this.indices);
        this.indices = newIndices;
        const newBitmap = this.allocator.acquire(Uint32Array, bitmapWordCount(newCapacity));
        newBitmap.set(this.nullBitmap);
        this.nullBitmap = newBitmap;
        this.capacity = newCapacity;
      }
    };
  }
});

// src/storage/chunk.js
var DEFAULT_CHUNK_SIZE, DataChunk;
var init_chunk = __esm({
  "src/storage/chunk.js"() {
    init_buffer_shim();
    init_column();
    init_dictionary_column();
    init_data_type();
    DEFAULT_CHUNK_SIZE = 2048;
    DataChunk = class _DataChunk {
      constructor(columns, size = 0) {
        this.columns = columns;
        this.size = size;
        this.selectionVector = null;
        this._cachedColumnCount = columns.length;
      }
      static fromSchema(schema, capacity = DEFAULT_CHUNK_SIZE) {
        const columns = schema.map(({ dataType }) => {
          if (dataType === DataType.VARCHAR) {
            return new DictionaryColumn(capacity);
          }
          return new Column(dataType, capacity);
        });
        return new _DataChunk(columns, 0);
      }
      getColumn(index) {
        return this.columns[index];
      }
      columnCount() {
        return this.columns.length;
      }
      getValue(rowIndex, colIndex) {
        const actualRow = this.selectionVector ? this.selectionVector[rowIndex] : rowIndex;
        return this.columns[colIndex].get(actualRow);
      }
      activeRowIndex(i) {
        return this.selectionVector ? this.selectionVector[i] : i;
      }
      setSelectionVector(sv, count2) {
        this.selectionVector = sv;
        this.size = count2;
      }
      clearSelectionVector() {
        this.selectionVector = null;
      }
      appendRow(values) {
        const rowIdx = this.size;
        for (let i = 0; i < values.length; i++) {
          this.columns[i].set(rowIdx, values[i]);
        }
        this.size++;
      }
      project(indices) {
        const projectedColumns = indices.map((i) => this.columns[i]);
        const chunk = new _DataChunk(projectedColumns, this.size);
        chunk.selectionVector = this.selectionVector;
        return chunk;
      }
      flatten() {
        if (!this.selectionVector) return this;
        const newColumns = this.columns.map((col2) => {
          const newCol = new Column(col2.dataType, this.size);
          for (let i = 0; i < this.size; i++) {
            newCol.set(i, col2.get(this.selectionVector[i]));
          }
          newCol.length = this.size;
          return newCol;
        });
        return new _DataChunk(newColumns, this.size);
      }
      reset() {
        this.size = 0;
        this.selectionVector = null;
        for (const col2 of this.columns) {
          col2.length = 0;
        }
      }
      toRows() {
        const rows = [];
        for (let i = 0; i < this.size; i++) {
          const row = [];
          for (let j = 0; j < this.columns.length; j++) {
            row.push(this.getValue(i, j));
          }
          rows.push(row);
        }
        return rows;
      }
    };
  }
});

// src/runtime/platform.js
function getEnvInt(key, fallback) {
  const val = processEnv?.[key];
  return val !== void 0 ? parseInt(val, 10) : fallback;
}
function getEnvFloat(key, fallback) {
  const val = processEnv?.[key];
  return val !== void 0 ? parseFloat(val) : fallback;
}
function getEnvFlag(key, fallback) {
  const val = processEnv?.[key];
  if (val === void 0) return fallback;
  return val === "1" || val.toLowerCase() === "true";
}
function getCpuCount() {
  return globalThis.navigator?.hardwareConcurrency ?? SINGLE_THREADED;
}
var SINGLE_THREADED, processEnv;
var init_platform = __esm({
  "src/runtime/platform.js"() {
    init_buffer_shim();
    SINGLE_THREADED = 1;
    processEnv = typeof process !== "undefined" && process.env ? process.env : null;
  }
});

// src/config.js
var config_exports = {};
__export(config_exports, {
  Config: () => Config
});
var env, envFlag, envFloat, resolveWorkerCount, Config;
var init_config = __esm({
  "src/config.js"() {
    init_buffer_shim();
    init_chunk();
    init_platform();
    env = getEnvInt;
    envFlag = getEnvFlag;
    envFloat = getEnvFloat;
    resolveWorkerCount = () => {
      const raw = env("QE_PARALLEL_WORKERS", 0);
      if (raw > 0) return raw;
      return Math.max(1, getCpuCount() - 1);
    };
    Config = {
      memoryLimit: env("QE_MEMORY_LIMIT", 2e5),
      hashJoinPartitions: env("QE_HASH_JOIN_PARTITIONS", 16),
      flushBatchSize: env("QE_FLUSH_BATCH_SIZE", DEFAULT_CHUNK_SIZE),
      bufferPoolPages: env("QE_BUFFER_POOL_PAGES", 50),
      wasmMinChunkSize: env("QE_WASM_MIN_CHUNK", 4096),
      sinkQueueCapacity: env("QE_SINK_QUEUE_CAPACITY", 8),
      btreeOrder: env("QE_BTREE_ORDER", 128),
      indexScanSelectivityThreshold: envFloat("QE_INDEX_SELECTIVITY_THRESHOLD", 0.3),
      dependentJoinConcurrency: env("QE_DEPENDENT_JOIN_CONCURRENCY", 1),
      parallelWorkers: resolveWorkerCount(),
      parallelThreshold: env("QE_PARALLEL_THRESHOLD", 1e4),
      parallelAggThreshold: env("QE_PARALLEL_AGG_THRESHOLD", 5e4),
      aggMorselRows: env("QE_AGG_MORSEL_ROWS", 16384),
      sabColumns: envFlag("QE_SAB_COLUMNS", false),
      sabArenaSegmentBytes: env("QE_SAB_ARENA_SEGMENT_BYTES", 1 << 20),
      parallelAggMemoryBytes: env("QE_PARALLEL_AGG_MEMORY_BYTES", 1 << 28),
      parallelCombineMinGroups: env("QE_PARALLEL_COMBINE_MIN_GROUPS", 8192),
      aggSpillGroups: env("QE_AGG_SPILL_GROUPS", 1 << 17),
      vectorGroupRange: env("QE_VECTOR_GROUP_RANGE", 1 << 21),
      parallelJoinThreshold: env("QE_PARALLEL_JOIN_THRESHOLD", 5e4),
      transportMaxBuffers: env("QE_TRANSPORT_MAX_BUFFERS", 64),
      aggRadixMultiplier: env("QE_AGG_RADIX_MULTIPLIER", 2),
      regionSize: env("QE_WASM_REGION_SIZE", 16 * 1024 * 1024),
      morselSize: env("QE_MORSEL_SIZE", 262144),
      clusterPort: env("QE_CLUSTER_PORT", 9400),
      coordinatorSchemaSampleRows: env("QE_COORD_SCHEMA_SAMPLE_ROWS", 1e3),
      heartbeatIntervalMs: env("QE_HEARTBEAT_INTERVAL", 3e3),
      heartbeatTimeoutMs: env("QE_HEARTBEAT_TIMEOUT", 1e4),
      defaultPartitionCount: env("QE_PARTITION_COUNT", 16),
      broadcastThreshold: env("QE_BROADCAST_THRESHOLD", 1e4),
      exchangeBatchSize: env("QE_EXCHANGE_BATCH_SIZE", 4096),
      exchangeBufferCapacity: env("QE_EXCHANGE_BUFFER_CAPACITY", 8),
      fragmentRetryLimit: env("QE_FRAGMENT_RETRY_LIMIT", 3),
      coordinatorTimeoutMs: env("QE_COORDINATOR_TIMEOUT", 3e5),
      codecCompression: env("QE_CODEC_COMPRESSION", 0),
      distributedWorkers: env("QE_DISTRIBUTED_WORKERS", 0),
      phiAccrualWindowSize: env("QE_PHI_WINDOW_SIZE", 100),
      phiAccrualThreshold: envFloat("QE_PHI_THRESHOLD", 8),
      networkCostPerByte: envFloat("QE_NETWORK_COST_PER_BYTE", 1e-3)
    };
  }
});

// src/storage/sab-arena.js
var HeapAllocator, heapAllocator;
var init_sab_arena = __esm({
  "src/storage/sab-arena.js"() {
    init_buffer_shim();
    init_config();
    HeapAllocator = class {
      get shared() {
        return false;
      }
      acquire(Ctor, length) {
        return new Ctor(length);
      }
    };
    heapAllocator = new HeapAllocator();
  }
});

// src/storage/column.js
var DEFAULT_CAPACITY2, STRING_INITIAL_BYTES, Column;
var init_column = __esm({
  "src/storage/column.js"() {
    init_buffer_shim();
    init_data_type();
    init_bitmap();
    init_sab_arena();
    DEFAULT_CAPACITY2 = 2048;
    STRING_INITIAL_BYTES = 4096;
    Column = class _Column {
      constructor(dataType, capacity = DEFAULT_CAPACITY2, allocator = heapAllocator) {
        this.dataType = dataType;
        this.capacity = capacity;
        this.length = 0;
        this.allocator = allocator;
        if (isFixedWidth(dataType)) {
          this.data = allocator.acquire(typedArrayCtorFor(dataType), capacity);
        } else {
          this.offsets = allocator.acquire(Uint32Array, capacity + 1);
          this.stringBytes = allocator.acquire(Uint8Array, STRING_INITIAL_BYTES);
          this.stringBytesUsed = 0;
        }
        this.nullBitmap = allocator.acquire(Uint32Array, bitmapWordCount(capacity));
        this.hasNulls = false;
      }
      static fromParts({ dataType, data, offsets, stringBytes, stringBytesUsed, nullBitmap, length, hasNulls, allocator }) {
        const col2 = Object.create(_Column.prototype);
        col2.dataType = dataType;
        col2.capacity = length;
        col2.length = length;
        col2.allocator = allocator || heapAllocator;
        if (data !== void 0) col2.data = data;
        if (offsets !== void 0) {
          col2.offsets = offsets;
          col2.stringBytes = stringBytes;
          col2.stringBytesUsed = stringBytesUsed;
        }
        col2.nullBitmap = nullBitmap;
        col2.hasNulls = hasNulls;
        return col2;
      }
      get(index) {
        if (this.hasNulls && !testBit(this.nullBitmap, index)) {
          return null;
        }
        if (this.dataType === DataType.VARCHAR) {
          return this._getString(index);
        }
        const val = this.data[index];
        if (this.dataType === DataType.BOOLEAN) {
          return val !== 0;
        }
        return val;
      }
      set(index, value) {
        if (value === null || value === void 0) {
          this._setNull(index);
          return;
        }
        setBit(this.nullBitmap, index);
        if (this.dataType === DataType.VARCHAR) {
          this._setString(index, value);
        } else if (this.dataType === DataType.BOOLEAN) {
          this.data[index] = value ? 1 : 0;
        } else {
          this.data[index] = value;
        }
        if (index >= this.length) {
          this.length = index + 1;
        }
      }
      append(value) {
        if (this.length >= this.capacity) {
          this._grow();
        }
        this.set(this.length, value);
      }
      appendBatch(values) {
        for (let i = 0; i < values.length; i++) {
          this.append(values[i]);
        }
      }
      isNull(index) {
        return this.hasNulls && !testBit(this.nullBitmap, index);
      }
      slice(start, end) {
        const len = end - start;
        const col2 = new _Column(this.dataType, len, this.allocator);
        if (this.dataType === DataType.VARCHAR) {
          for (let i = 0; i < len; i++) {
            col2.set(i, this.get(start + i));
          }
        } else {
          if (this.data) {
            const srcSlice = this.data.subarray(start, end);
            col2.data.set(srcSlice);
          }
          for (let i = 0; i < len; i++) {
            if (testBit(this.nullBitmap, start + i)) {
              setBit(col2.nullBitmap, i);
            }
          }
        }
        col2.length = len;
        col2.hasNulls = this.hasNulls;
        return col2;
      }
      _setNull(index) {
        this.hasNulls = true;
        clearBit(this.nullBitmap, index);
        if (this.dataType === DataType.VARCHAR) {
          this.offsets[index + 1] = this.offsets[index];
        }
        if (index >= this.length) {
          this.length = index + 1;
        }
      }
      _getString(index) {
        const start = this.offsets[index];
        const end = this.offsets[index + 1];
        if (start === end) return "";
        const bytes = this.stringBytes.subarray(start, end);
        return new TextDecoder().decode(bytes);
      }
      _setString(index, value) {
        const encoded = new TextEncoder().encode(value);
        while (this.stringBytesUsed + encoded.length > this.stringBytes.length) {
          this._growStringBuffer();
        }
        this.offsets[index] = this.stringBytesUsed;
        this.stringBytes.set(encoded, this.stringBytesUsed);
        this.stringBytesUsed += encoded.length;
        this.offsets[index + 1] = this.stringBytesUsed;
      }
      _grow() {
        const newCapacity = this.capacity * 2;
        if (isFixedWidth(this.dataType)) {
          const newData = this.allocator.acquire(typedArrayCtorFor(this.dataType), newCapacity);
          newData.set(this.data);
          this.data = newData;
        } else {
          const newOffsets = this.allocator.acquire(Uint32Array, newCapacity + 1);
          newOffsets.set(this.offsets);
          this.offsets = newOffsets;
        }
        const newBitmap = this.allocator.acquire(Uint32Array, bitmapWordCount(newCapacity));
        newBitmap.set(this.nullBitmap);
        this.nullBitmap = newBitmap;
        this.capacity = newCapacity;
      }
      _growStringBuffer() {
        const newBuffer = this.allocator.acquire(Uint8Array, this.stringBytes.length * 2);
        newBuffer.set(this.stringBytes);
        this.stringBytes = newBuffer;
      }
    };
  }
});

// src/wasm/dispatch.js
var dispatch_exports = {};
__export(dispatch_exports, {
  WasmDispatch: () => WasmDispatch,
  globalDispatch: () => globalDispatch
});
var WasmDispatch, globalDispatch;
var init_dispatch = __esm({
  "src/wasm/dispatch.js"() {
    init_buffer_shim();
    WasmDispatch = class {
      constructor() {
        this.kernels = /* @__PURE__ */ new Map();
        this.wasmInstance = null;
        this.memory = null;
      }
      register(operation, dataType, kernel) {
        const key = `${operation}:${dataType}`;
        this.kernels.set(key, kernel);
      }
      lookup(operation, dataType) {
        const key = `${operation}:${dataType}`;
        return this.kernels.get(key) || null;
      }
      has(operation, dataType) {
        return this.kernels.has(`${operation}:${dataType}`);
      }
      setInstance(instance, memory) {
        this.wasmInstance = instance;
        this.memory = memory;
      }
      listKernels() {
        return [...this.kernels.keys()];
      }
    };
    globalDispatch = new WasmDispatch();
  }
});

// src/planner/plan-formatter.js
var plan_formatter_exports = {};
__export(plan_formatter_exports, {
  formatExpression: () => formatExpression,
  formatPlan: () => formatPlan
});
function formatExpression(expr2) {
  if (!expr2) return "";
  switch (expr2.kind) {
    case BoundExprKind.COLUMN_REF:
      return expr2.tableAlias ? `${expr2.tableAlias}.${expr2.columnName}` : expr2.columnName;
    case BoundExprKind.LITERAL:
      if (typeof expr2.value === "string") return `'${expr2.value}'`;
      return String(expr2.value);
    case BoundExprKind.BINARY:
      return `(${formatExpression(expr2.left)} ${expr2.op} ${formatExpression(expr2.right)})`;
    case BoundExprKind.UNARY:
      return `${expr2.op} ${formatExpression(expr2.operand)}`;
    case BoundExprKind.FUNCTION:
      return `${expr2.name}(${expr2.args.map(formatExpression).join(", ")})`;
    case BoundExprKind.AGGREGATE:
      return `${expr2.name}(${expr2.args.map(formatExpression).join(", ")})`;
    default:
      return expr2.kind ? `<${expr2.kind}>` : JSON.stringify(expr2);
  }
}
function formatPlan(plan, depth = 0) {
  const indent = "  ".repeat(depth);
  let result = `${indent}-> ${formatNode(plan)}
`;
  if (plan.children && plan.children.length > 0) {
    for (const child of plan.children) {
      result += formatPlan(child, depth + 1);
    }
  }
  return result;
}
function formatNode(node) {
  switch (node.type) {
    case PlanNodeType.SCAN:
      return `Seq Scan on ${node.table}${node.alias ? " as " + node.alias : ""}`;
    case PlanNodeType.FILTER:
      return `Filter (condition: ${formatExpression(node.condition)})`;
    case PlanNodeType.PROJECT:
      const exprs = node.expressions.map((e) => formatExpression(e)).join(", ");
      return `Project (${exprs})`;
    case PlanNodeType.JOIN: {
      let joinTypeStr = node.joinType === "INNER" ? "" : `${node.joinType} `;
      let physicalPrefix = "";
      if (node.physicalStrategy === "HASH") physicalPrefix = "Hash ";
      else if (node.physicalStrategy === "MERGE") physicalPrefix = "Merge ";
      let joinStr = `${physicalPrefix}${joinTypeStr}Join`;
      if (node.condition) joinStr += ` (condition: ${formatExpression(node.condition)})`;
      return joinStr;
    }
    case PlanNodeType.AGGREGATE: {
      let physicalPrefix = "";
      if (node.physicalStrategy === "HASH") physicalPrefix = "Hash ";
      else if (node.physicalStrategy === "STREAM") physicalPrefix = "Stream ";
      else if (node.physicalStrategy === "PERFECT_HASH") physicalPrefix = "Perfect Hash ";
      else if (node.physicalStrategy === "UNGROUPED") physicalPrefix = "Ungrouped ";
      let aggStr = `${physicalPrefix}Aggregate`;
      if (node.groupBy && node.groupBy.length > 0) {
        aggStr += ` (group by: ${node.groupBy.map((g) => formatExpression(g)).join(", ")})`;
      }
      if (node.aggregates && node.aggregates.length > 0) {
        aggStr += ` (aggs: ${node.aggregates.map((a) => `${a.name}(${a.args.map((arg) => formatExpression(arg)).join(", ")})`).join(", ")})`;
      }
      return aggStr;
    }
    case PlanNodeType.SORT:
      const sorts = node.orderKeys.map((k) => `${formatExpression(k.expr)} ${k.direction || "ASC"}`).join(", ");
      return `Sort (${sorts})`;
    case PlanNodeType.LIMIT:
      return `Limit (count: ${node.count}${node.offset ? ", offset: " + node.offset : ""})`;
    case PlanNodeType.TOP_N: {
      const topNSorts = node.orderKeys.map((k) => `${formatExpression(k.expr)} ${k.direction || "ASC"}`).join(", ");
      return `Top-N (count: ${node.count}${node.offset ? ", offset: " + node.offset : ""}, order: ${topNSorts})`;
    }
    case PlanNodeType.DISTINCT:
      return `Distinct`;
    case PlanNodeType.UNION:
      return `Union${node.all ? " All" : ""}`;
    case PlanNodeType.CTE_ANCHOR:
      return `CTE Anchor (${node.cteName})`;
    case PlanNodeType.CTE_SCAN:
      return `CTE Scan (${node.cteName})`;
    case PlanNodeType.MATERIALIZE:
      return `Materialize`;
    case PlanNodeType.DEPENDENT_JOIN:
      return `Dependent Join (${node.subqueryType})`;
    case PlanNodeType.INDEX_SCAN: {
      let desc = `Index Scan using ${node.indexName} on ${node.table}`;
      if (node.alias !== node.table) desc += ` as ${node.alias}`;
      if (node.scanType === "point") desc += ` (key: ${node.scanKey})`;
      else desc += ` (range: ${node.scanLow ?? "-\u221E"} to ${node.scanHigh ?? "\u221E"})`;
      return desc;
    }
    case PlanNodeType.EMPTY:
      return `Empty (short-circuit)`;
    case PlanNodeType.WINDOW: {
      const wExprs = node.windowExprs.map((w) => w.name).join(", ");
      return `Window (${wExprs})`;
    }
    default:
      return `${node.type}`;
  }
}
var init_plan_formatter = __esm({
  "src/planner/plan-formatter.js"() {
    init_buffer_shim();
    init_logical_plan();
    init_expression_binder();
  }
});

// src/wasm/region-allocator.js
var ALIGNMENT, align, RegionAllocator;
var init_region_allocator = __esm({
  "src/wasm/region-allocator.js"() {
    init_buffer_shim();
    ALIGNMENT = 16;
    align = (offset) => offset + ALIGNMENT - 1 & ~(ALIGNMENT - 1);
    RegionAllocator = class {
      constructor(memory, regionSize) {
        this.memory = memory;
        this.regionSize = regionSize;
        this.regions = [];
        this.totalAllocated = 0;
        this.stagingOffset = 0;
        this.dataOffset = 0;
        this.dataBaseOffset = 0;
      }
      addRegion() {
        const id = this.regions.length;
        const start = this.totalAllocated;
        const capacity = this.regionSize;
        this._ensureMemory(start + capacity);
        this.regions.push({
          id,
          start,
          capacity,
          offset: 0
        });
        this.totalAllocated = start + capacity;
        return id;
      }
      alloc(regionId, bytes) {
        const region = this.regions[regionId];
        if (!region) throw new RangeError(`Region ${regionId} does not exist`);
        const alignedOffset = align(region.offset);
        const needed = alignedOffset + bytes;
        if (needed > region.capacity) {
          this._growRegion(regionId, needed);
        }
        const ptr = region.start + alignedOffset;
        region.offset = alignedOffset + bytes;
        return ptr;
      }
      reset(regionId) {
        const region = this.regions[regionId];
        if (!region) throw new RangeError(`Region ${regionId} does not exist`);
        region.offset = 0;
      }
      getRegionBounds(regionId) {
        const region = this.regions[regionId];
        if (!region) throw new RangeError(`Region ${regionId} does not exist`);
        return { start: region.start, end: region.start + region.capacity };
      }
      getRegionUsage(regionId) {
        const region = this.regions[regionId];
        if (!region) throw new RangeError(`Region ${regionId} does not exist`);
        return { used: region.offset, capacity: region.capacity, free: region.capacity - region.offset };
      }
      regionCount() {
        return this.regions.length;
      }
      _growRegion(regionId, needed) {
        const region = this.regions[regionId];
        const isLast = regionId === this.regions.length - 1;
        if (!isLast) {
          throw new RangeError(
            `Region ${regionId} exhausted (${region.capacity} bytes) and cannot grow \u2014 only the last region is growable`
          );
        }
        let newCapacity = region.capacity;
        while (newCapacity < needed) {
          newCapacity *= 2;
        }
        this._ensureMemory(region.start + newCapacity);
        const growth = newCapacity - region.capacity;
        region.capacity = newCapacity;
        this.totalAllocated += growth;
      }
      allocData(bytes) {
        if (this.dataBaseOffset === 0) {
          this.dataBaseOffset = this.totalAllocated;
        }
        const alignedOffset = align(this.dataOffset);
        const ptr = this.dataBaseOffset + alignedOffset;
        const needed = ptr + bytes;
        this._ensureMemory(needed);
        this.dataOffset = alignedOffset + bytes;
        return ptr;
      }
      resetData() {
        this.dataOffset = 0;
      }
      allocStaging(bytes) {
        const base = this.dataBaseOffset > 0 ? this.dataBaseOffset + this.dataOffset : this.totalAllocated;
        const alignedOffset = align(this.stagingOffset);
        const ptr = base + alignedOffset;
        const needed = ptr + bytes;
        this._ensureMemory(needed);
        this.stagingOffset = alignedOffset + bytes;
        return ptr;
      }
      resetStaging() {
        this.stagingOffset = 0;
      }
      _ensureMemory(requiredBytes) {
        const currentBytes = this.memory.buffer.byteLength;
        if (requiredBytes <= currentBytes) return;
        const PAGE_SIZE = 65536;
        const pagesNeeded = Math.ceil((requiredBytes - currentBytes) / PAGE_SIZE);
        this.memory.grow(pagesNeeded);
      }
    };
  }
});

// src/wasm/wasm-loader-base.js
var WASM_PAGE_SIZE, INITIAL_PAGES, MAX_PAGES, ALIGNMENT2, BYTE_WIDTH_I32, BYTE_WIDTH_F64, WasmLoaderBase;
var init_wasm_loader_base = __esm({
  "src/wasm/wasm-loader-base.js"() {
    init_buffer_shim();
    init_region_allocator();
    WASM_PAGE_SIZE = 65536;
    INITIAL_PAGES = 256;
    MAX_PAGES = 16384;
    ALIGNMENT2 = 16;
    BYTE_WIDTH_I32 = 4;
    BYTE_WIDTH_F64 = 8;
    WasmLoaderBase = class {
      constructor() {
        this.memory = null;
        this.instances = /* @__PURE__ */ new Map();
        this.modules = /* @__PURE__ */ new Map();
        this.moduleBytes = /* @__PURE__ */ new Map();
        this.bumpOffset = 0;
        this.regionAllocator = null;
        this.shared = false;
      }
      initRegions(regionSize) {
        if (!this.memory) throw new Error("Memory not initialized \u2014 call init() first");
        this.regionAllocator = new RegionAllocator(this.memory, regionSize);
        return this.regionAllocator;
      }
      async fetchModuleBytes(_name) {
        throw new Error("fetchModuleBytes must be implemented by a platform loader");
      }
      async loadModule(name) {
        if (this.instances.has(name)) return this.instances.get(name);
        if (!this.memory) await this.init();
        const buffer = await this.fetchModuleBytes(name);
        this.moduleBytes.set(name, buffer);
        const module = await WebAssembly.compile(buffer);
        this.modules.set(name, module);
        const imports = WebAssembly.Module.imports(module);
        const needsMemoryImport = imports.some((i) => i.name === "memory");
        const instance = await WebAssembly.instantiate(module, {
          env: needsMemoryImport ? { memory: this.memory } : {}
        });
        if (!needsMemoryImport && instance.exports.memory) {
          this.memory = instance.exports.memory;
        }
        this.instances.set(name, instance);
        return instance;
      }
      getModule(name) {
        return this.modules.get(name) || null;
      }
      getModuleBytes(name) {
        return this.moduleBytes.get(name) || null;
      }
      alloc(bytes) {
        const aligned = this.bumpOffset + ALIGNMENT2 - 1 & ~(ALIGNMENT2 - 1);
        const paddedBytes = bytes + ALIGNMENT2;
        const newOffset = aligned + paddedBytes;
        const totalBytes = this.memory.buffer.byteLength;
        if (newOffset > totalBytes) {
          const pagesNeeded = Math.ceil((newOffset - totalBytes) / WASM_PAGE_SIZE);
          this.memory.grow(pagesNeeded);
        }
        this.bumpOffset = newOffset;
        return aligned;
      }
      reset() {
        this.bumpOffset = 0;
      }
      getBuffer() {
        return this.memory.buffer;
      }
      isShared() {
        return this.shared;
      }
      isWasmBacked(data) {
        return data.buffer === this.memory.buffer;
      }
      allocTypedArray(TypedArrayCtor, length) {
        if (!this.regionAllocator) return null;
        const bw = TypedArrayCtor.BYTES_PER_ELEMENT;
        const ptr = this.regionAllocator.allocData(length * bw);
        return new TypedArrayCtor(this.memory.buffer, ptr, length);
      }
      resolveDataPtr(data, byteWidth) {
        if (this.isWasmBacked(data)) return data.byteOffset;
        const bytes = data.length * byteWidth;
        const ptr = this.alloc(bytes);
        if (byteWidth === BYTE_WIDTH_I32) this.writeI32Array(data, ptr);
        else this.writeF64Array(data, ptr);
        return ptr;
      }
      writeI32Array(data, ptr) {
        new Int32Array(this.memory.buffer, ptr, data.length).set(data);
      }
      writeF64Array(data, ptr) {
        new Float64Array(this.memory.buffer, ptr, data.length).set(data);
      }
      readI32Array(ptr, length) {
        return new Int32Array(this.memory.buffer.slice(ptr, ptr + length * BYTE_WIDTH_I32));
      }
      readF64Array(ptr, length) {
        return new Float64Array(this.memory.buffer.slice(ptr, ptr + length * BYTE_WIDTH_F64));
      }
      readF64(ptr) {
        return new Float64Array(this.memory.buffer, ptr, 1)[0];
      }
      readU32Array(ptr, length) {
        return new Uint32Array(this.memory.buffer.slice(ptr, ptr + length * BYTE_WIDTH_I32));
      }
    };
  }
});

// src/wasm/loader.js
var loader_exports = {};
__export(loader_exports, {
  WasmLoader: () => WasmLoader,
  configureWasmSource: () => configureWasmSource,
  getGlobalLoader: () => getGlobalLoader,
  resetGlobalLoader: () => resetGlobalLoader
});
function configureWasmSource(byteSource) {
  _byteSource = byteSource;
}
async function getGlobalLoader(options = {}) {
  if (!_globalLoader) {
    if (!_byteSource) {
      throw new Error("WASM byte source not configured \u2014 call configureWasmSource() first");
    }
    _globalLoader = new WasmLoader(_byteSource);
    await _globalLoader.init(options);
  }
  return _globalLoader;
}
function resetGlobalLoader() {
  _globalLoader = null;
}
var WasmLoader, _globalLoader, _byteSource;
var init_loader = __esm({
  "src/wasm/loader.js"() {
    init_buffer_shim();
    init_wasm_loader_base();
    WasmLoader = class extends WasmLoaderBase {
      constructor(byteSource) {
        super();
        this.byteSource = byteSource;
      }
      async init(_options = {}) {
        this.shared = true;
        this.memory = new WebAssembly.Memory({
          initial: INITIAL_PAGES,
          maximum: MAX_PAGES,
          shared: true
        });
        this.bumpOffset = 0;
      }
      async fetchModuleBytes(name) {
        return this.byteSource(name);
      }
    };
    _globalLoader = null;
    _byteSource = null;
  }
});

// src/wasm/kernels/wasm-filter.js
async function getCoreInstance() {
  if (_coreInstance) return _coreInstance;
  const loader = await getGlobalLoader();
  _coreInstance = { loader, instance: await loader.loadModule("core") };
  return _coreInstance;
}
async function wasmFilterEqI32(data, value) {
  const { loader, instance } = await getCoreInstance();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.resolveDataPtr(data, 4);
  const selVecPtr = loader.alloc(count2 * 4);
  const matchCount = instance.exports.filterEqI32(dataPtr, selVecPtr, count2, value);
  return loader.readU32Array(selVecPtr, matchCount);
}
async function wasmFilterLtI32(data, value) {
  const { loader, instance } = await getCoreInstance();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.resolveDataPtr(data, 4);
  const selVecPtr = loader.alloc(count2 * 4);
  const matchCount = instance.exports.filterLtI32(dataPtr, selVecPtr, count2, value);
  return loader.readU32Array(selVecPtr, matchCount);
}
async function wasmFilterGtI32(data, value) {
  const { loader, instance } = await getCoreInstance();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.resolveDataPtr(data, 4);
  const selVecPtr = loader.alloc(count2 * 4);
  const matchCount = instance.exports.filterGtI32(dataPtr, selVecPtr, count2, value);
  return loader.readU32Array(selVecPtr, matchCount);
}
async function wasmFilterBetweenI32(data, low, high) {
  const { loader, instance } = await getCoreInstance();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.resolveDataPtr(data, 4);
  const selVecPtr = loader.alloc(count2 * 4);
  const matchCount = instance.exports.filterBetweenI32(dataPtr, selVecPtr, count2, low, high);
  return loader.readU32Array(selVecPtr, matchCount);
}
var _coreInstance;
var init_wasm_filter = __esm({
  "src/wasm/kernels/wasm-filter.js"() {
    init_buffer_shim();
    init_loader();
    _coreInstance = null;
  }
});

// src/wasm/kernels/wasm-filter-f64.js
async function getCoreInstance2() {
  if (_coreInstance2) return _coreInstance2;
  const loader = await getGlobalLoader();
  _coreInstance2 = { loader, instance: await loader.loadModule("core") };
  return _coreInstance2;
}
async function wasmFilterEqF64(data, value) {
  const { loader, instance } = await getCoreInstance2();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.resolveDataPtr(data, 8);
  const selVecPtr = loader.alloc(count2 * 4);
  const matchCount = instance.exports.filterEqF64(dataPtr, selVecPtr, count2, value);
  return loader.readU32Array(selVecPtr, matchCount);
}
async function wasmFilterLtF64(data, value) {
  const { loader, instance } = await getCoreInstance2();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.resolveDataPtr(data, 8);
  const selVecPtr = loader.alloc(count2 * 4);
  const matchCount = instance.exports.filterLtF64(dataPtr, selVecPtr, count2, value);
  return loader.readU32Array(selVecPtr, matchCount);
}
async function wasmFilterGtF64(data, value) {
  const { loader, instance } = await getCoreInstance2();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.resolveDataPtr(data, 8);
  const selVecPtr = loader.alloc(count2 * 4);
  const matchCount = instance.exports.filterGtF64(dataPtr, selVecPtr, count2, value);
  return loader.readU32Array(selVecPtr, matchCount);
}
async function wasmFilterLeF64(data, value) {
  const { loader, instance } = await getCoreInstance2();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.resolveDataPtr(data, 8);
  const selVecPtr = loader.alloc(count2 * 4);
  const matchCount = instance.exports.filterLeF64(dataPtr, selVecPtr, count2, value);
  return loader.readU32Array(selVecPtr, matchCount);
}
async function wasmFilterGeF64(data, value) {
  const { loader, instance } = await getCoreInstance2();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.resolveDataPtr(data, 8);
  const selVecPtr = loader.alloc(count2 * 4);
  const matchCount = instance.exports.filterGeF64(dataPtr, selVecPtr, count2, value);
  return loader.readU32Array(selVecPtr, matchCount);
}
async function wasmFilterBetweenF64(data, low, high) {
  const { loader, instance } = await getCoreInstance2();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.resolveDataPtr(data, 8);
  const selVecPtr = loader.alloc(count2 * 4);
  const matchCount = instance.exports.filterBetweenF64(dataPtr, selVecPtr, count2, low, high);
  return loader.readU32Array(selVecPtr, matchCount);
}
async function wasmFilterLeI32(data, value) {
  const { loader, instance } = await getCoreInstance2();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.resolveDataPtr(data, 4);
  const selVecPtr = loader.alloc(count2 * 4);
  const matchCount = instance.exports.filterLeI32(dataPtr, selVecPtr, count2, value);
  return loader.readU32Array(selVecPtr, matchCount);
}
async function wasmFilterGeI32(data, value) {
  const { loader, instance } = await getCoreInstance2();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.resolveDataPtr(data, 4);
  const selVecPtr = loader.alloc(count2 * 4);
  const matchCount = instance.exports.filterGeI32(dataPtr, selVecPtr, count2, value);
  return loader.readU32Array(selVecPtr, matchCount);
}
var _coreInstance2;
var init_wasm_filter_f64 = __esm({
  "src/wasm/kernels/wasm-filter-f64.js"() {
    init_buffer_shim();
    init_loader();
    _coreInstance2 = null;
  }
});

// src/wasm/kernels/wasm-aggregate.js
async function getCoreInstance3() {
  if (_coreInstance3) return _coreInstance3;
  const loader = await getGlobalLoader();
  _coreInstance3 = { loader, instance: await loader.loadModule("core") };
  return _coreInstance3;
}
async function wasmSumI32(data) {
  const { loader, instance } = await getCoreInstance3();
  loader.reset();
  const dataPtr = loader.resolveDataPtr(data, 4);
  return instance.exports.sumI32(dataPtr, data.length);
}
async function wasmSumF64(data) {
  const { loader, instance } = await getCoreInstance3();
  loader.reset();
  const dataPtr = loader.resolveDataPtr(data, 8);
  return instance.exports.sumF64(dataPtr, data.length);
}
async function wasmMinI32(data) {
  const { loader, instance } = await getCoreInstance3();
  loader.reset();
  const dataPtr = loader.resolveDataPtr(data, 4);
  return instance.exports.minI32(dataPtr, data.length);
}
async function wasmMaxI32(data) {
  const { loader, instance } = await getCoreInstance3();
  loader.reset();
  const dataPtr = loader.resolveDataPtr(data, 4);
  return instance.exports.maxI32(dataPtr, data.length);
}
async function wasmMinF64(data) {
  const { loader, instance } = await getCoreInstance3();
  loader.reset();
  const dataPtr = loader.resolveDataPtr(data, 8);
  return instance.exports.minF64(dataPtr, data.length);
}
async function wasmMaxF64(data) {
  const { loader, instance } = await getCoreInstance3();
  loader.reset();
  const dataPtr = loader.resolveDataPtr(data, 8);
  return instance.exports.maxF64(dataPtr, data.length);
}
var _coreInstance3;
var init_wasm_aggregate = __esm({
  "src/wasm/kernels/wasm-aggregate.js"() {
    init_buffer_shim();
    init_loader();
    _coreInstance3 = null;
  }
});

// src/wasm/kernels/wasm-arithmetic.js
async function getCoreInstance4() {
  if (_coreInstance4) return _coreInstance4;
  const loader = await getGlobalLoader();
  _coreInstance4 = { loader, instance: await loader.loadModule("core") };
  return _coreInstance4;
}
async function wasmVecAddF64(a, b) {
  const { loader, instance } = await getCoreInstance4();
  loader.reset();
  const count2 = a.length;
  const aPtr = loader.alloc(count2 * 8);
  const bPtr = loader.alloc(count2 * 8);
  const outPtr = loader.alloc(count2 * 8);
  loader.writeF64Array(a, aPtr);
  loader.writeF64Array(b, bPtr);
  instance.exports.vecAddF64(aPtr, bPtr, outPtr, count2);
  return loader.readF64Array(outPtr, count2);
}
async function wasmVecSubF64(a, b) {
  const { loader, instance } = await getCoreInstance4();
  loader.reset();
  const count2 = a.length;
  const aPtr = loader.alloc(count2 * 8);
  const bPtr = loader.alloc(count2 * 8);
  const outPtr = loader.alloc(count2 * 8);
  loader.writeF64Array(a, aPtr);
  loader.writeF64Array(b, bPtr);
  instance.exports.vecSubF64(aPtr, bPtr, outPtr, count2);
  return loader.readF64Array(outPtr, count2);
}
async function wasmVecMulF64(a, b) {
  const { loader, instance } = await getCoreInstance4();
  loader.reset();
  const count2 = a.length;
  const aPtr = loader.alloc(count2 * 8);
  const bPtr = loader.alloc(count2 * 8);
  const outPtr = loader.alloc(count2 * 8);
  loader.writeF64Array(a, aPtr);
  loader.writeF64Array(b, bPtr);
  instance.exports.vecMulF64(aPtr, bPtr, outPtr, count2);
  return loader.readF64Array(outPtr, count2);
}
async function wasmVecDivF64(a, b) {
  const { loader, instance } = await getCoreInstance4();
  loader.reset();
  const count2 = a.length;
  const aPtr = loader.alloc(count2 * 8);
  const bPtr = loader.alloc(count2 * 8);
  const outPtr = loader.alloc(count2 * 8);
  loader.writeF64Array(a, aPtr);
  loader.writeF64Array(b, bPtr);
  instance.exports.vecDivF64(aPtr, bPtr, outPtr, count2);
  return loader.readF64Array(outPtr, count2);
}
async function wasmScalarAddF64(data, scalar) {
  const { loader, instance } = await getCoreInstance4();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.alloc(count2 * 8);
  const outPtr = loader.alloc(count2 * 8);
  loader.writeF64Array(data, dataPtr);
  instance.exports.scalarAddF64(dataPtr, scalar, outPtr, count2);
  return loader.readF64Array(outPtr, count2);
}
async function wasmScalarSubF64(data, scalar) {
  const { loader, instance } = await getCoreInstance4();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.alloc(count2 * 8);
  const outPtr = loader.alloc(count2 * 8);
  loader.writeF64Array(data, dataPtr);
  instance.exports.scalarSubF64(dataPtr, scalar, outPtr, count2);
  return loader.readF64Array(outPtr, count2);
}
async function wasmScalarMulF64(data, scalar) {
  const { loader, instance } = await getCoreInstance4();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.alloc(count2 * 8);
  const outPtr = loader.alloc(count2 * 8);
  loader.writeF64Array(data, dataPtr);
  instance.exports.scalarMulF64(dataPtr, scalar, outPtr, count2);
  return loader.readF64Array(outPtr, count2);
}
async function wasmScalarDivF64(data, scalar) {
  const { loader, instance } = await getCoreInstance4();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.alloc(count2 * 8);
  const outPtr = loader.alloc(count2 * 8);
  loader.writeF64Array(data, dataPtr);
  instance.exports.scalarDivF64(dataPtr, scalar, outPtr, count2);
  return loader.readF64Array(outPtr, count2);
}
async function wasmScalarSubRevF64(scalar, data) {
  const { loader, instance } = await getCoreInstance4();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.alloc(count2 * 8);
  const outPtr = loader.alloc(count2 * 8);
  loader.writeF64Array(data, dataPtr);
  instance.exports.scalarSubRevF64(scalar, dataPtr, outPtr, count2);
  return loader.readF64Array(outPtr, count2);
}
async function wasmScalarDivRevF64(scalar, data) {
  const { loader, instance } = await getCoreInstance4();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.alloc(count2 * 8);
  const outPtr = loader.alloc(count2 * 8);
  loader.writeF64Array(data, dataPtr);
  instance.exports.scalarDivRevF64(scalar, dataPtr, outPtr, count2);
  return loader.readF64Array(outPtr, count2);
}
async function wasmWidenI32ToF64(data) {
  const { loader, instance } = await getCoreInstance4();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.alloc(count2 * 4);
  const outPtr = loader.alloc(count2 * 8);
  loader.writeI32Array(data, dataPtr);
  instance.exports.widenI32ToF64(dataPtr, outPtr, count2);
  return loader.readF64Array(outPtr, count2);
}
async function wasmNegF64(data) {
  const { loader, instance } = await getCoreInstance4();
  loader.reset();
  const count2 = data.length;
  const dataPtr = loader.alloc(count2 * 8);
  const outPtr = loader.alloc(count2 * 8);
  loader.writeF64Array(data, dataPtr);
  instance.exports.negF64(dataPtr, outPtr, count2);
  return loader.readF64Array(outPtr, count2);
}
async function wasmCountBits(bitmap, bitCount) {
  const { loader, instance } = await getCoreInstance4();
  loader.reset();
  const wordCount = Math.ceil(bitCount / 32);
  const byteCount = wordCount * 4;
  const bitmapPtr = loader.alloc(byteCount);
  new Uint8Array(loader.getBuffer(), bitmapPtr, byteCount).set(
    new Uint8Array(bitmap.buffer, bitmap.byteOffset, byteCount)
  );
  return instance.exports.countBitmapBits(bitmapPtr, bitCount);
}
var _coreInstance4;
var init_wasm_arithmetic = __esm({
  "src/wasm/kernels/wasm-arithmetic.js"() {
    init_buffer_shim();
    init_loader();
    _coreInstance4 = null;
  }
});

// src/wasm/register-kernels.js
var register_kernels_exports = {};
__export(register_kernels_exports, {
  registerAllKernels: () => registerAllKernels
});
function registerAllKernels() {
  globalDispatch.register("filterEq", "INT32", wasmFilterEqI32);
  globalDispatch.register("filterLt", "INT32", wasmFilterLtI32);
  globalDispatch.register("filterGt", "INT32", wasmFilterGtI32);
  globalDispatch.register("filterLe", "INT32", wasmFilterLeI32);
  globalDispatch.register("filterGe", "INT32", wasmFilterGeI32);
  globalDispatch.register("filterBetween", "INT32", wasmFilterBetweenI32);
  globalDispatch.register("filterEq", "FLOAT64", wasmFilterEqF64);
  globalDispatch.register("filterLt", "FLOAT64", wasmFilterLtF64);
  globalDispatch.register("filterGt", "FLOAT64", wasmFilterGtF64);
  globalDispatch.register("filterLe", "FLOAT64", wasmFilterLeF64);
  globalDispatch.register("filterGe", "FLOAT64", wasmFilterGeF64);
  globalDispatch.register("filterBetween", "FLOAT64", wasmFilterBetweenF64);
  globalDispatch.register("filterEq", "DATE", wasmFilterEqI32);
  globalDispatch.register("filterLt", "DATE", wasmFilterLtI32);
  globalDispatch.register("filterGt", "DATE", wasmFilterGtI32);
  globalDispatch.register("filterLe", "DATE", wasmFilterLeI32);
  globalDispatch.register("filterGe", "DATE", wasmFilterGeI32);
  globalDispatch.register("filterBetween", "DATE", wasmFilterBetweenI32);
  globalDispatch.register("sumI32", "INT32", wasmSumI32);
  globalDispatch.register("sumF64", "FLOAT64", wasmSumF64);
  globalDispatch.register("minI32", "INT32", wasmMinI32);
  globalDispatch.register("maxI32", "INT32", wasmMaxI32);
  globalDispatch.register("minF64", "FLOAT64", wasmMinF64);
  globalDispatch.register("maxF64", "FLOAT64", wasmMaxF64);
  globalDispatch.register("vecAddF64", "FLOAT64", wasmVecAddF64);
  globalDispatch.register("vecSubF64", "FLOAT64", wasmVecSubF64);
  globalDispatch.register("vecMulF64", "FLOAT64", wasmVecMulF64);
  globalDispatch.register("vecDivF64", "FLOAT64", wasmVecDivF64);
  globalDispatch.register("scalarAddF64", "FLOAT64", wasmScalarAddF64);
  globalDispatch.register("scalarSubF64", "FLOAT64", wasmScalarSubF64);
  globalDispatch.register("scalarMulF64", "FLOAT64", wasmScalarMulF64);
  globalDispatch.register("scalarDivF64", "FLOAT64", wasmScalarDivF64);
  globalDispatch.register("scalarSubRevF64", "FLOAT64", wasmScalarSubRevF64);
  globalDispatch.register("scalarDivRevF64", "FLOAT64", wasmScalarDivRevF64);
  globalDispatch.register("widenI32ToF64", "INT32", wasmWidenI32ToF64);
  globalDispatch.register("negF64", "FLOAT64", wasmNegF64);
  globalDispatch.register("countBits", "UINT8", wasmCountBits);
}
var init_register_kernels = __esm({
  "src/wasm/register-kernels.js"() {
    init_buffer_shim();
    init_dispatch();
    init_wasm_filter();
    init_wasm_filter_f64();
    init_wasm_aggregate();
    init_wasm_arithmetic();
  }
});

// src/browser.js
init_buffer_shim();

// src/engine/query-engine.js
init_buffer_shim();

// src/parser/parser.js
init_buffer_shim();

// src/parser/lexer.js
init_buffer_shim();
var TokenType = {
  IDENT: "IDENT",
  NUMBER: "NUMBER",
  STRING: "STRING",
  COMMA: "COMMA",
  DOT: "DOT",
  STAR: "STAR",
  LPAREN: "LPAREN",
  RPAREN: "RPAREN",
  EQ: "EQ",
  NEQ: "NEQ",
  LT: "LT",
  GT: "GT",
  LTE: "LTE",
  GTE: "GTE",
  PLUS: "PLUS",
  MINUS: "MINUS",
  SLASH: "SLASH",
  PERCENT: "PERCENT",
  SEMICOLON: "SEMICOLON",
  CONCAT: "CONCAT",
  COLON: "COLON",
  EOF: "EOF",
  EXPLAIN: "EXPLAIN",
  SELECT: "SELECT",
  FROM: "FROM",
  WHERE: "WHERE",
  AND: "AND",
  OR: "OR",
  NOT: "NOT",
  AS: "AS",
  ON: "ON",
  JOIN: "JOIN",
  INNER: "INNER",
  LEFT: "LEFT",
  RIGHT: "RIGHT",
  FULL: "FULL",
  OUTER: "OUTER",
  CROSS: "CROSS",
  IN: "IN",
  EXISTS: "EXISTS",
  BETWEEN: "BETWEEN",
  LIKE: "LIKE",
  IS: "IS",
  NULL: "NULL",
  TRUE: "TRUE",
  FALSE: "FALSE",
  CASE: "CASE",
  WHEN: "WHEN",
  THEN: "THEN",
  ELSE: "ELSE",
  END: "END",
  DISTINCT: "DISTINCT",
  ALL: "ALL",
  GROUP: "GROUP",
  BY: "BY",
  HAVING: "HAVING",
  ORDER: "ORDER",
  ASC: "ASC",
  DESC: "DESC",
  LIMIT: "LIMIT",
  OFFSET: "OFFSET",
  UNION: "UNION",
  EXCEPT: "EXCEPT",
  INTERSECT: "INTERSECT",
  WITH: "WITH",
  CAST: "CAST",
  INTERVAL: "INTERVAL",
  EXTRACT: "EXTRACT",
  SUBSTRING: "SUBSTRING",
  TRIM: "TRIM",
  YEAR: "YEAR",
  MONTH: "MONTH",
  DAY: "DAY",
  DATE: "DATE",
  SUM: "SUM",
  AVG: "AVG",
  COUNT: "COUNT",
  MIN: "MIN",
  MAX: "MAX",
  CREATE: "CREATE",
  VIEW: "VIEW",
  NULLS: "NULLS",
  FIRST: "FIRST",
  LAST: "LAST",
  FETCH: "FETCH",
  NEXT: "NEXT",
  ROWS: "ROWS",
  ONLY: "ONLY",
  SOME: "SOME",
  ANY: "ANY",
  LEADING: "LEADING",
  TRAILING: "TRAILING",
  BOTH: "BOTH",
  FOR: "FOR",
  TIMESTAMP: "TIMESTAMP",
  HOUR: "HOUR",
  MINUTE: "MINUTE",
  SECOND: "SECOND",
  OVER: "OVER",
  PARTITION: "PARTITION",
  RANGE: "RANGE",
  UNBOUNDED: "UNBOUNDED",
  PRECEDING: "PRECEDING",
  FOLLOWING: "FOLLOWING",
  CURRENT: "CURRENT",
  ROW: "ROW",
  NATURAL: "NATURAL",
  USING: "USING",
  TABLE: "TABLE",
  DROP: "DROP",
  IF: "IF",
  ANALYZE: "ANALYZE"
};
var NON_KEYWORD_TOKENS = /* @__PURE__ */ new Set([
  "IDENT",
  "NUMBER",
  "STRING",
  "COMMA",
  "DOT",
  "STAR",
  "LPAREN",
  "RPAREN",
  "EQ",
  "NEQ",
  "LT",
  "GT",
  "LTE",
  "GTE",
  "PLUS",
  "MINUS",
  "SLASH",
  "PERCENT",
  "SEMICOLON",
  "CONCAT",
  "COLON",
  "EOF"
]);
var KEYWORDS = /* @__PURE__ */ new Map();
for (const key of Object.keys(TokenType)) {
  if (!NON_KEYWORD_TOKENS.has(key)) {
    KEYWORDS.set(key, TokenType[key]);
  }
}
var Token = class {
  constructor(type, value, position) {
    this.type = type;
    this.value = value;
    this.position = position;
  }
};
var Lexer = class {
  constructor(input) {
    this.input = input;
    this.pos = 0;
    this.tokens = [];
    this._tokenize();
  }
  _tokenize() {
    while (this.pos < this.input.length) {
      this._skipWhitespaceAndComments();
      if (this.pos >= this.input.length) break;
      const start = this.pos;
      const ch = this.input[this.pos];
      if (ch === "'") {
        this.tokens.push(this._readString(start));
      } else if (this._isDigit(ch)) {
        this.tokens.push(this._readNumber(start));
      } else if (this._isIdentStart(ch)) {
        this.tokens.push(this._readIdentOrKeyword(start));
      } else {
        this.tokens.push(this._readSymbol(start));
      }
    }
    this.tokens.push(new Token(TokenType.EOF, "", this.pos));
  }
  _skipWhitespaceAndComments() {
    while (this.pos < this.input.length) {
      const ch = this.input[this.pos];
      if (ch === " " || ch === "	" || ch === "\n" || ch === "\r") {
        this.pos++;
        continue;
      }
      if (ch === "-" && this.pos + 1 < this.input.length && this.input[this.pos + 1] === "-") {
        while (this.pos < this.input.length && this.input[this.pos] !== "\n") {
          this.pos++;
        }
        continue;
      }
      break;
    }
  }
  _readString(start) {
    this.pos++;
    let value = "";
    while (this.pos < this.input.length) {
      if (this.input[this.pos] === "'") {
        if (this.pos + 1 < this.input.length && this.input[this.pos + 1] === "'") {
          value += "'";
          this.pos += 2;
        } else {
          this.pos++;
          return new Token(TokenType.STRING, value, start);
        }
      } else {
        value += this.input[this.pos];
        this.pos++;
      }
    }
    throw new Error(`Unterminated string at position ${start}`);
  }
  _readNumber(start) {
    while (this.pos < this.input.length && this._isDigit(this.input[this.pos])) {
      this.pos++;
    }
    if (this.pos < this.input.length && this.input[this.pos] === ".") {
      this.pos++;
      while (this.pos < this.input.length && this._isDigit(this.input[this.pos])) {
        this.pos++;
      }
    }
    return new Token(TokenType.NUMBER, this.input.slice(start, this.pos), start);
  }
  _readIdentOrKeyword(start) {
    while (this.pos < this.input.length && this._isIdentPart(this.input[this.pos])) {
      this.pos++;
    }
    const value = this.input.slice(start, this.pos);
    const upper = value.toUpperCase();
    const keywordType = KEYWORDS.get(upper);
    if (keywordType) {
      return new Token(keywordType, upper, start);
    }
    return new Token(TokenType.IDENT, value, start);
  }
  _readSymbol(start) {
    const ch = this.input[this.pos];
    this.pos++;
    switch (ch) {
      case ",":
        return new Token(TokenType.COMMA, ",", start);
      case ".":
        return new Token(TokenType.DOT, ".", start);
      case "*":
        return new Token(TokenType.STAR, "*", start);
      case "(":
        return new Token(TokenType.LPAREN, "(", start);
      case ")":
        return new Token(TokenType.RPAREN, ")", start);
      case "+":
        return new Token(TokenType.PLUS, "+", start);
      case "-":
        return new Token(TokenType.MINUS, "-", start);
      case "/":
        return new Token(TokenType.SLASH, "/", start);
      case "%":
        return new Token(TokenType.PERCENT, "%", start);
      case ";":
        return new Token(TokenType.SEMICOLON, ";", start);
      case ":":
        return new Token(TokenType.COLON, ":", start);
      case "=":
        return new Token(TokenType.EQ, "=", start);
      case "<":
        if (this.pos < this.input.length) {
          if (this.input[this.pos] === "=") {
            this.pos++;
            return new Token(TokenType.LTE, "<=", start);
          }
          if (this.input[this.pos] === ">") {
            this.pos++;
            return new Token(TokenType.NEQ, "<>", start);
          }
        }
        return new Token(TokenType.LT, "<", start);
      case ">":
        if (this.pos < this.input.length && this.input[this.pos] === "=") {
          this.pos++;
          return new Token(TokenType.GTE, ">=", start);
        }
        return new Token(TokenType.GT, ">", start);
      case "!":
        if (this.pos < this.input.length && this.input[this.pos] === "=") {
          this.pos++;
          return new Token(TokenType.NEQ, "!=", start);
        }
        throw new Error(`Unexpected character '!' at position ${start}`);
      case "|":
        if (this.pos < this.input.length && this.input[this.pos] === "|") {
          this.pos++;
          return new Token(TokenType.CONCAT, "||", start);
        }
        throw new Error(`Unexpected character '|' at position ${start}`);
      default:
        throw new Error(`Unexpected character '${ch}' at position ${start}`);
    }
  }
  _isDigit(ch) {
    return ch >= "0" && ch <= "9";
  }
  _isIdentStart(ch) {
    return ch >= "a" && ch <= "z" || ch >= "A" && ch <= "Z" || ch === "_";
  }
  _isIdentPart(ch) {
    return this._isIdentStart(ch) || this._isDigit(ch);
  }
};

// src/parser/parser.js
init_ast();
var Parser = class {
  constructor(sql) {
    const lexer = new Lexer(sql);
    this.tokens = lexer.tokens;
    this.pos = 0;
  }
  parse() {
    if (this.isAt(TokenType.CREATE)) {
      return this.parseCreateTable();
    }
    if (this.isAt(TokenType.DROP)) {
      return this.parseDropTable();
    }
    let isExplain = false;
    let isAnalyze = false;
    if (this.isAt(TokenType.EXPLAIN)) {
      this.advance();
      isExplain = true;
      if (this.isAt(TokenType.ANALYZE)) {
        this.advance();
        isAnalyze = true;
      }
    }
    const stmt = this.parseQueryExpr();
    if (!this.isAt(TokenType.EOF) && !this.isAt(TokenType.SEMICOLON)) {
      this.error(`Unexpected token ${this.peek().type}`);
    }
    if (isAnalyze) return ExplainAnalyzeStmt(stmt);
    return isExplain ? ExplainStmt(stmt) : stmt;
  }
  parseCreateTable() {
    this.advance();
    this.expect(TokenType.TABLE);
    let ifNotExists = false;
    if (this.isAt(TokenType.IF)) {
      this.advance();
      this.expect(TokenType.NOT);
      this.expect(TokenType.EXISTS);
      ifNotExists = true;
    }
    const name = this.expectIdent();
    this.expect(TokenType.LPAREN);
    const columns = [];
    do {
      const colName = this.expectIdent();
      const colType = this.parseTypeName();
      columns.push(ColumnDef(colName, colType));
    } while (this.tryConsume(TokenType.COMMA));
    this.expect(TokenType.RPAREN);
    return CreateTableStmt(name, columns, ifNotExists);
  }
  parseDropTable() {
    this.advance();
    this.expect(TokenType.TABLE);
    let ifExists = false;
    if (this.isAt(TokenType.IF)) {
      this.advance();
      this.expect(TokenType.EXISTS);
      ifExists = true;
    }
    const name = this.expectIdent();
    return DropTableStmt(name, ifExists);
  }
  parseQueryExpr() {
    let left = this.parseSelectStmt();
    while (this.isAt(TokenType.UNION) || this.isAt(TokenType.EXCEPT) || this.isAt(TokenType.INTERSECT)) {
      const op = this.advance().type;
      const all = this.tryConsume(TokenType.ALL);
      if (!all) this.tryConsume(TokenType.DISTINCT);
      const right = this.parseSelectStmt();
      left = SetOp(op, left, right, all);
    }
    return left;
  }
  parseSelectStmt() {
    let withClause = null;
    if (this.isAt(TokenType.WITH)) {
      withClause = this.parseWithClause();
    }
    if (this.isAt(TokenType.LPAREN)) {
      this.advance();
      const inner = this.parseQueryExpr();
      this.expect(TokenType.RPAREN);
      return inner;
    }
    this.expect(TokenType.SELECT);
    const distinct = this.tryConsume(TokenType.DISTINCT);
    if (!distinct) this.tryConsume(TokenType.ALL);
    const selectItems = this.parseSelectItems();
    let from = null;
    if (this.tryConsume(TokenType.FROM)) {
      from = this.parseFromClause();
    }
    let where = null;
    if (this.tryConsume(TokenType.WHERE)) {
      where = this.parseExpression();
    }
    let groupBy = null;
    if (this.tryConsume(TokenType.GROUP)) {
      this.expect(TokenType.BY);
      groupBy = this.parseExpressionList();
    }
    let having = null;
    if (this.tryConsume(TokenType.HAVING)) {
      having = this.parseExpression();
    }
    let orderBy = null;
    if (this.isAt(TokenType.ORDER)) {
      this.advance();
      this.expect(TokenType.BY);
      orderBy = this.parseOrderByList();
    }
    let limit = null;
    let offset = null;
    if (this.tryConsume(TokenType.LIMIT)) {
      limit = this.parseExpression();
      if (this.tryConsume(TokenType.OFFSET)) {
        offset = this.parseExpression();
      }
    }
    if (this.tryConsume(TokenType.FETCH)) {
      if (!this.tryConsume(TokenType.FIRST)) {
        this.tryConsume(TokenType.NEXT);
      }
      limit = this.parseExpression();
      this.tryConsume(TokenType.ROWS);
      this.expect(TokenType.ONLY);
    }
    return SelectStmt({ withClause, distinct: !!distinct, selectItems, from, where, groupBy, having, orderBy, limit, offset });
  }
  parseWithClause() {
    this.advance();
    const ctes = [];
    do {
      ctes.push(this.parseCTE());
    } while (this.tryConsume(TokenType.COMMA));
    return WithClause(ctes);
  }
  parseCTE() {
    const name = this.expectIdent();
    let columnAliases = null;
    if (this.isAt(TokenType.LPAREN) && !this.isLookaheadSelect()) {
      this.advance();
      columnAliases = [];
      do {
        columnAliases.push(this.expectIdent());
      } while (this.tryConsume(TokenType.COMMA));
      this.expect(TokenType.RPAREN);
    }
    this.expect(TokenType.AS);
    this.expect(TokenType.LPAREN);
    const query = this.parseQueryExpr();
    this.expect(TokenType.RPAREN);
    return CTE(name, query, columnAliases);
  }
  parseSelectItems() {
    const items = [];
    do {
      items.push(this.parseSelectItem());
    } while (this.tryConsume(TokenType.COMMA));
    return items;
  }
  parseSelectItem() {
    if (this.isAt(TokenType.STAR)) {
      this.advance();
      return SelectItem(AllColumns());
    }
    if (this.isAt(TokenType.IDENT) && this.peekAhead(1)?.type === TokenType.DOT && this.peekAhead(2)?.type === TokenType.STAR) {
      const table = this.advance().value;
      this.advance();
      this.advance();
      return SelectItem(AllColumns(table));
    }
    const expr2 = this.parseExpression();
    let alias = null;
    if (this.tryConsume(TokenType.AS)) {
      alias = this.expectIdent();
    } else if (this.isAt(TokenType.IDENT)) {
      alias = this.expectIdent();
    }
    return SelectItem(expr2, alias);
  }
  parseFromClause() {
    let left = this.parseTableRef();
    while (this.isJoinKeyword()) {
      left = this.parseJoin(left);
    }
    while (this.tryConsume(TokenType.COMMA)) {
      const right = this.parseTableRef();
      left = JoinRef(left, right, "CROSS");
    }
    return left;
  }
  parseTableRef() {
    if (this.isAt(TokenType.LPAREN)) {
      if (this.isSubqueryStart()) {
        return this.parseSubqueryRef();
      }
      this.advance();
      const inner = this.parseFromClause();
      this.expect(TokenType.RPAREN);
      return inner;
    }
    const name = this.expectIdent();
    let alias = name;
    if (this.tryConsume(TokenType.AS)) {
      alias = this.expectIdent();
    } else if (this.isAt(TokenType.IDENT) && !this.isJoinKeyword() && !this.isClauseKeyword()) {
      alias = this.expectIdent();
    }
    return TableRef(name, alias);
  }
  parseSubqueryRef() {
    this.expect(TokenType.LPAREN);
    const query = this.parseQueryExpr();
    this.expect(TokenType.RPAREN);
    let alias = null;
    if (this.tryConsume(TokenType.AS)) {
      alias = this.expectIdent();
    } else if (this.isAt(TokenType.IDENT)) {
      alias = this.expectIdent();
    }
    return SubqueryRef(query, alias);
  }
  parseJoin(left) {
    let joinType = "INNER";
    let isNatural = false;
    if (this.tryConsume(TokenType.NATURAL)) {
      isNatural = true;
    }
    if (this.tryConsume(TokenType.LEFT)) {
      this.tryConsume(TokenType.OUTER);
      joinType = "LEFT";
    } else if (this.tryConsume(TokenType.RIGHT)) {
      this.tryConsume(TokenType.OUTER);
      joinType = "RIGHT";
    } else if (this.tryConsume(TokenType.FULL)) {
      this.tryConsume(TokenType.OUTER);
      joinType = "FULL";
    } else if (this.tryConsume(TokenType.CROSS)) {
      joinType = "CROSS";
    } else if (this.tryConsume(TokenType.INNER)) {
      joinType = "INNER";
    }
    this.expect(TokenType.JOIN);
    const right = this.parseTableRef();
    let condition = null;
    let usingColumns = null;
    if (isNatural) {
      const node2 = JoinRef(left, right, joinType, null);
      node2.natural = true;
      return node2;
    }
    if (joinType !== "CROSS") {
      if (this.tryConsume(TokenType.ON)) {
        condition = this.parseExpression();
      } else if (this.tryConsume(TokenType.USING)) {
        this.expect(TokenType.LPAREN);
        usingColumns = [];
        do {
          usingColumns.push(this.expectIdent());
        } while (this.tryConsume(TokenType.COMMA));
        this.expect(TokenType.RPAREN);
      }
    }
    const node = JoinRef(left, right, joinType, condition);
    if (usingColumns) node.usingColumns = usingColumns;
    return node;
  }
  parseExpression() {
    return this.parseOr();
  }
  parseOr() {
    let left = this.parseAnd();
    while (this.isAt(TokenType.OR)) {
      this.advance();
      const right = this.parseAnd();
      left = BinaryExpr("OR", left, right);
    }
    return left;
  }
  parseAnd() {
    let left = this.parseNot();
    while (this.isAt(TokenType.AND)) {
      this.advance();
      const right = this.parseNot();
      left = BinaryExpr("AND", left, right);
    }
    return left;
  }
  parseNot() {
    if (this.isAt(TokenType.NOT)) {
      this.advance();
      const operand = this.parseNot();
      return UnaryExpr("NOT", operand);
    }
    return this.parseComparison();
  }
  parseComparison() {
    if (this.isAt(TokenType.EXISTS)) {
      return this.parseExists();
    }
    let left = this.parseAddition();
    if (this.isAt(TokenType.NOT)) {
      const saved = this.pos;
      this.advance();
      if (this.isAt(TokenType.BETWEEN)) {
        return this.parseBetween(left, true);
      } else if (this.isAt(TokenType.IN)) {
        return this.parseIn(left, true);
      } else if (this.isAt(TokenType.LIKE)) {
        return this.parseLike(left, true);
      }
      this.pos = saved;
    }
    if (this.isAt(TokenType.BETWEEN)) return this.parseBetween(left, false);
    if (this.isAt(TokenType.IN)) return this.parseIn(left, false);
    if (this.isAt(TokenType.LIKE)) return this.parseLike(left, false);
    if (this.isAt(TokenType.IS)) {
      this.advance();
      const negated = this.tryConsume(TokenType.NOT);
      this.expect(TokenType.NULL);
      return IsNullExpr(left, !!negated);
    }
    const opMap = {
      [TokenType.EQ]: "=",
      [TokenType.NEQ]: "<>",
      [TokenType.LT]: "<",
      [TokenType.GT]: ">",
      [TokenType.LTE]: "<=",
      [TokenType.GTE]: ">="
    };
    if (opMap[this.peek().type]) {
      const op = opMap[this.advance().type];
      if (this.isAt(TokenType.ALL) || this.isAt(TokenType.SOME) || this.isAt(TokenType.ANY)) {
        const quantifier = this.advance().type;
        this.expect(TokenType.LPAREN);
        const query = this.parseQueryExpr();
        this.expect(TokenType.RPAREN);
        return BinaryExpr(op, left, { kind: "QuantifiedSubquery", quantifier, query });
      }
      const right = this.parseAddition();
      return BinaryExpr(op, left, right);
    }
    return left;
  }
  parseBetween(left, negated) {
    this.advance();
    const low = this.parseAddition();
    this.expect(TokenType.AND);
    const high = this.parseAddition();
    return BetweenExpr(left, low, high, negated);
  }
  parseIn(left, negated) {
    this.advance();
    this.expect(TokenType.LPAREN);
    if (this.isAt(TokenType.SELECT) || this.isAt(TokenType.WITH)) {
      const query = this.parseQueryExpr();
      this.expect(TokenType.RPAREN);
      return InExpr(left, SubqueryExpr(query), negated);
    }
    const list = this.parseExpressionList();
    this.expect(TokenType.RPAREN);
    return InExpr(left, list, negated);
  }
  parseLike(left, negated) {
    this.advance();
    const pattern = this.parseAddition();
    return LikeExpr(left, pattern, negated);
  }
  parseExists() {
    this.advance();
    this.expect(TokenType.LPAREN);
    const query = this.parseQueryExpr();
    this.expect(TokenType.RPAREN);
    return ExistsExpr(query, false);
  }
  parseAddition() {
    let left = this.parseMultiplication();
    while (this.isAt(TokenType.PLUS) || this.isAt(TokenType.MINUS) || this.isAt(TokenType.CONCAT)) {
      const op = this.advance().value;
      const right = this.parseMultiplication();
      left = BinaryExpr(op, left, right);
    }
    return left;
  }
  parseMultiplication() {
    let left = this.parseUnary();
    while (this.isAt(TokenType.STAR) || this.isAt(TokenType.SLASH) || this.isAt(TokenType.PERCENT)) {
      const op = this.advance().value;
      const right = this.parseUnary();
      left = BinaryExpr(op, left, right);
    }
    return left;
  }
  parseUnary() {
    if (this.isAt(TokenType.MINUS)) {
      this.advance();
      const operand = this.parseUnary();
      return UnaryExpr("-", operand);
    }
    if (this.isAt(TokenType.PLUS)) {
      this.advance();
      return this.parseUnary();
    }
    return this.parsePrimary();
  }
  parsePrimary() {
    const token = this.peek();
    if (token.type === TokenType.NUMBER) {
      this.advance();
      return Literal(token.value.includes(".") ? parseFloat(token.value) : parseInt(token.value, 10));
    }
    if (token.type === TokenType.STRING) {
      this.advance();
      return Literal(token.value, "VARCHAR");
    }
    if (token.type === TokenType.NULL) {
      this.advance();
      return Literal(null);
    }
    if (token.type === TokenType.TRUE) {
      this.advance();
      return Literal(true, "BOOLEAN");
    }
    if (token.type === TokenType.FALSE) {
      this.advance();
      return Literal(false, "BOOLEAN");
    }
    if (token.type === TokenType.DATE) {
      this.advance();
      const dateStr = this.expect(TokenType.STRING);
      return Literal(dateStr.value, "DATE");
    }
    if (token.type === TokenType.TIMESTAMP) {
      this.advance();
      const tsStr = this.expect(TokenType.STRING);
      return Literal(tsStr.value, "TIMESTAMP");
    }
    if (token.type === TokenType.INTERVAL) {
      return this.parseInterval();
    }
    if (token.type === TokenType.CASE) {
      return this.parseCaseExpr();
    }
    if (token.type === TokenType.CAST) {
      return this.parseCast();
    }
    if (token.type === TokenType.EXTRACT) {
      return this.parseExtract();
    }
    if (token.type === TokenType.SUBSTRING) {
      return this.parseSubstringFn();
    }
    if (token.type === TokenType.TRIM) {
      return this.parseTrim();
    }
    if (token.type === TokenType.EXISTS) {
      return this.parseExists();
    }
    if (token.type === TokenType.NOT) {
      this.advance();
      if (this.isAt(TokenType.EXISTS)) {
        this.advance();
        this.expect(TokenType.LPAREN);
        const query = this.parseQueryExpr();
        this.expect(TokenType.RPAREN);
        return ExistsExpr(query, true);
      }
      const operand = this.parsePrimary();
      return UnaryExpr("NOT", operand);
    }
    if (token.type === TokenType.LPAREN) {
      if (this.isSubqueryStart()) {
        this.advance();
        const query = this.parseQueryExpr();
        this.expect(TokenType.RPAREN);
        return SubqueryExpr(query);
      }
      this.advance();
      const expr2 = this.parseExpression();
      this.expect(TokenType.RPAREN);
      return expr2;
    }
    if (token.type === TokenType.STAR) {
      this.advance();
      return AllColumns();
    }
    if (this.isAggregateKeyword(token.type)) {
      return this.parseAggregateCall();
    }
    if (token.type === TokenType.COLON) {
      this.advance();
      const paramToken = this.expect(TokenType.NUMBER);
      return Literal(`:${paramToken.value}`, "PARAM");
    }
    if (token.type === TokenType.IDENT) {
      const name = this.advance().value;
      if (this.isAt(TokenType.LPAREN)) {
        return this.parseFunctionCallNamed(name);
      }
      if (this.isAt(TokenType.DOT)) {
        this.advance();
        if (this.isAt(TokenType.STAR)) {
          this.advance();
          return AllColumns(name);
        }
        const colName = this.expectIdent();
        return ColumnRef(colName, name);
      }
      return ColumnRef(name);
    }
    this.error(`Unexpected token ${token.type} (${token.value})`);
  }
  parseAggregateCall() {
    const name = this.advance().value;
    this.expect(TokenType.LPAREN);
    if (name === "COUNT" && this.isAt(TokenType.STAR)) {
      this.advance();
      this.expect(TokenType.RPAREN);
      if (this.isAt(TokenType.OVER)) {
        return this.parseWindowCall("COUNT_STAR", []);
      }
      return AggregateCall("COUNT_STAR", [], false);
    }
    const distinct = this.tryConsume(TokenType.DISTINCT);
    const args = this.parseExpressionList();
    this.expect(TokenType.RPAREN);
    if (this.isAt(TokenType.OVER)) {
      return this.parseWindowCall(name, args);
    }
    return AggregateCall(name, args, !!distinct);
  }
  parseFunctionCallNamed(name) {
    this.expect(TokenType.LPAREN);
    let args = [];
    if (!this.isAt(TokenType.RPAREN)) {
      args = this.parseExpressionList();
    }
    this.expect(TokenType.RPAREN);
    if (this.isAt(TokenType.OVER)) {
      return this.parseWindowCall(name, args);
    }
    return FunctionCall(name, args);
  }
  parseWindowCall(name, args) {
    this.expect(TokenType.OVER);
    this.expect(TokenType.LPAREN);
    let partitionBy = [];
    if (this.tryConsume(TokenType.PARTITION)) {
      this.expect(TokenType.BY);
      partitionBy = this.parseExpressionList();
    }
    let orderBy = [];
    if (this.isAt(TokenType.ORDER)) {
      this.advance();
      this.expect(TokenType.BY);
      orderBy = this.parseOrderByList();
    }
    this.expect(TokenType.RPAREN);
    const windowSpec = WindowSpec(partitionBy, orderBy);
    return WindowCall(name, args, windowSpec);
  }
  parseCaseExpr() {
    this.advance();
    let operand = null;
    if (!this.isAt(TokenType.WHEN)) {
      operand = this.parseExpression();
    }
    const whenClauses = [];
    while (this.tryConsume(TokenType.WHEN)) {
      const condition = this.parseExpression();
      this.expect(TokenType.THEN);
      const result = this.parseExpression();
      whenClauses.push(WhenClause(condition, result));
    }
    let elseExpr = null;
    if (this.tryConsume(TokenType.ELSE)) {
      elseExpr = this.parseExpression();
    }
    this.expect(TokenType.END);
    return CaseExpr(operand, whenClauses, elseExpr);
  }
  parseCast() {
    this.advance();
    this.expect(TokenType.LPAREN);
    const expr2 = this.parseExpression();
    this.expect(TokenType.AS);
    const targetType = this.parseTypeName();
    this.expect(TokenType.RPAREN);
    return CastExpr(expr2, targetType);
  }
  parseExtract() {
    this.advance();
    this.expect(TokenType.LPAREN);
    const field = this.peek().value;
    this.advance();
    this.expect(TokenType.FROM);
    const source = this.parseExpression();
    this.expect(TokenType.RPAREN);
    return ExtractExpr(field.toUpperCase(), source);
  }
  parseSubstringFn() {
    this.advance();
    this.expect(TokenType.LPAREN);
    const expr2 = this.parseExpression();
    let from, length = null;
    if (this.isAtKeyword("FROM")) {
      this.advance();
      from = this.parseExpression();
      if (this.isAtKeyword("FOR")) {
        this.advance();
        length = this.parseExpression();
      }
    } else {
      this.expect(TokenType.COMMA);
      from = this.parseExpression();
      if (this.tryConsume(TokenType.COMMA)) {
        length = this.parseExpression();
      }
    }
    this.expect(TokenType.RPAREN);
    return SubstringExpr(expr2, from, length);
  }
  parseTrim() {
    this.advance();
    this.expect(TokenType.LPAREN);
    let trimSpec = null;
    if (this.isAt(TokenType.LEADING) || this.isAt(TokenType.TRAILING) || this.isAt(TokenType.BOTH)) {
      trimSpec = this.advance().value;
    }
    let trimChar = null;
    if (!this.isAt(TokenType.FROM) && trimSpec !== null) {
      if (!this.isAt(TokenType.FROM)) {
        trimChar = this.parseExpression();
      }
    }
    if (trimSpec !== null) {
      this.tryConsume(TokenType.FROM);
    }
    const expr2 = this.parseExpression();
    this.expect(TokenType.RPAREN);
    const args = trimChar ? [expr2, Literal(trimSpec), trimChar] : [expr2];
    return FunctionCall("TRIM", args);
  }
  parseInterval() {
    this.advance();
    const valueToken = this.expect(TokenType.STRING);
    const unit = this.expectIdent().toUpperCase();
    return IntervalExpr(valueToken.value, unit);
  }
  parseTypeName() {
    const name = this.expectIdent().toUpperCase();
    const params = [];
    if (this.tryConsume(TokenType.LPAREN)) {
      do {
        params.push(parseInt(this.expect(TokenType.NUMBER).value, 10));
      } while (this.tryConsume(TokenType.COMMA));
      this.expect(TokenType.RPAREN);
    }
    return TypeName(name, params);
  }
  parseExpressionList() {
    const list = [];
    do {
      list.push(this.parseExpression());
    } while (this.tryConsume(TokenType.COMMA));
    return list;
  }
  parseOrderByList() {
    const list = [];
    do {
      const expr2 = this.parseExpression();
      let direction = "ASC";
      if (this.tryConsume(TokenType.ASC)) direction = "ASC";
      else if (this.tryConsume(TokenType.DESC)) direction = "DESC";
      let nullOrder = null;
      if (this.tryConsume(TokenType.NULLS)) {
        if (this.tryConsume(TokenType.FIRST)) nullOrder = "FIRST";
        else if (this.tryConsume(TokenType.LAST)) nullOrder = "LAST";
      }
      list.push(OrderKey(expr2, direction, nullOrder));
    } while (this.tryConsume(TokenType.COMMA));
    return list;
  }
  peek() {
    return this.tokens[this.pos];
  }
  peekAhead(offset) {
    return this.tokens[this.pos + offset] || null;
  }
  advance() {
    return this.tokens[this.pos++];
  }
  isAt(type) {
    return this.peek().type === type;
  }
  tryConsume(type) {
    if (this.isAt(type)) {
      this.advance();
      return true;
    }
    return false;
  }
  expect(type) {
    if (!this.isAt(type)) {
      this.error(`Expected ${type}, got ${this.peek().type} (${this.peek().value})`);
    }
    return this.advance();
  }
  expectIdent() {
    const token = this.peek();
    if (token.type === TokenType.IDENT) {
      this.advance();
      return token.value;
    }
    if (this.isNonReservedKeyword(token.type)) {
      this.advance();
      return token.value;
    }
    this.error(`Expected identifier, got ${token.type} (${token.value})`);
  }
  isNonReservedKeyword(type) {
    return [
      TokenType.YEAR,
      TokenType.MONTH,
      TokenType.DAY,
      TokenType.DATE,
      TokenType.FIRST,
      TokenType.LAST,
      TokenType.NULLS,
      TokenType.VIEW,
      TokenType.ROWS,
      TokenType.ONLY,
      TokenType.NEXT,
      TokenType.FETCH,
      TokenType.SOME,
      TokenType.ANY,
      TokenType.LEADING,
      TokenType.TRAILING,
      TokenType.BOTH,
      TokenType.SUM,
      TokenType.AVG,
      TokenType.COUNT,
      TokenType.MIN,
      TokenType.MAX,
      TokenType.SUBSTRING,
      TokenType.EXTRACT,
      TokenType.TRIM,
      TokenType.OFFSET,
      TokenType.FOR,
      TokenType.ASC,
      TokenType.DESC,
      TokenType.LEFT,
      TokenType.RIGHT,
      TokenType.FULL,
      TokenType.INNER,
      TokenType.OUTER,
      TokenType.CROSS,
      TokenType.JOIN,
      TokenType.CREATE,
      TokenType.INTERVAL,
      TokenType.UNION,
      TokenType.ALL,
      TokenType.TIMESTAMP,
      TokenType.HOUR,
      TokenType.MINUTE,
      TokenType.SECOND,
      TokenType.OVER,
      TokenType.PARTITION,
      TokenType.RANGE,
      TokenType.UNBOUNDED,
      TokenType.PRECEDING,
      TokenType.FOLLOWING,
      TokenType.CURRENT,
      TokenType.ROW,
      TokenType.NATURAL,
      TokenType.USING,
      TokenType.TABLE,
      TokenType.DROP,
      TokenType.IF,
      TokenType.ANALYZE
    ].includes(type);
  }
  isAggregateKeyword(type) {
    return [TokenType.SUM, TokenType.AVG, TokenType.COUNT, TokenType.MIN, TokenType.MAX].includes(type);
  }
  isJoinKeyword() {
    const type = this.peek().type;
    return type === TokenType.JOIN || type === TokenType.INNER || type === TokenType.LEFT || type === TokenType.RIGHT || type === TokenType.FULL || type === TokenType.CROSS || type === TokenType.NATURAL;
  }
  isClauseKeyword() {
    const type = this.peek().type;
    return type === TokenType.WHERE || type === TokenType.GROUP || type === TokenType.HAVING || type === TokenType.ORDER || type === TokenType.LIMIT || type === TokenType.UNION || type === TokenType.EXCEPT || type === TokenType.INTERSECT || type === TokenType.ON || type === TokenType.FETCH;
  }
  isSubqueryStart() {
    let depth = 0;
    for (let i = this.pos; i < this.tokens.length; i++) {
      if (this.tokens[i].type === TokenType.LPAREN) depth++;
      if (this.tokens[i].type === TokenType.RPAREN) depth--;
      if (depth === 0 && i > this.pos) break;
      if (depth === 1 && (this.tokens[i].type === TokenType.SELECT || this.tokens[i].type === TokenType.WITH)) {
        return true;
      }
    }
    return false;
  }
  isLookaheadSelect() {
    let depth = 1;
    for (let i = this.pos + 1; i < this.tokens.length; i++) {
      if (this.tokens[i].type === TokenType.LPAREN) depth++;
      if (this.tokens[i].type === TokenType.RPAREN) {
        depth--;
        if (depth === 0) break;
      }
      if (depth === 1 && this.tokens[i].type === TokenType.SELECT) return true;
    }
    return false;
  }
  isAtKeyword(keyword) {
    const t = this.peek();
    const upper = keyword.toUpperCase();
    return t.type === upper || t.value === upper;
  }
  error(message) {
    const token = this.peek();
    throw new Error(`Parse error at position ${token.position}: ${message}`);
  }
};
function parse(sql) {
  return new Parser(sql).parse();
}
function parseExpression(sql) {
  const parser = new Parser(sql);
  const expr2 = parser.parseExpression();
  if (!parser.isAt(TokenType.EOF) && !parser.isAt(TokenType.SEMICOLON)) {
    parser.error(`Unexpected token ${parser.peek().type}`);
  }
  return expr2;
}

// src/binder/binder.js
init_buffer_shim();
init_ast();
init_data_type();

// src/binder/scope.js
init_buffer_shim();
var BinderScope = class _BinderScope {
  constructor(parent = null) {
    this.parent = parent;
    this.tables = /* @__PURE__ */ new Map();
    this.columns = /* @__PURE__ */ new Map();
    this.correlatedRefs = [];
  }
  addTable(alias, tableInfo) {
    this.tables.set(alias.toUpperCase(), tableInfo);
  }
  addColumn(alias, columnInfo) {
    const key = alias.toUpperCase();
    if (!this.columns.has(key)) {
      this.columns.set(key, []);
    }
    this.columns.get(key).push(columnInfo);
  }
  resolveTable(name) {
    const upper = name.toUpperCase();
    const local = this.tables.get(upper);
    if (local) return { table: local, depth: 0 };
    if (this.parent) {
      const result = this.parent.resolveTable(upper);
      if (result) return { table: result.table, depth: result.depth + 1 };
    }
    return null;
  }
  resolveColumn(name, tableAlias = null) {
    const upper = name.toUpperCase();
    if (tableAlias) {
      const tableUpper = tableAlias.toUpperCase();
      const tableResult = this.resolveTable(tableAlias);
      if (!tableResult) return null;
      const { table, depth } = tableResult;
      const col2 = table.columns.find((c) => c.name.toUpperCase() === upper);
      if (!col2) return null;
      const colIndex = table.columns.findIndex((c) => c.name.toUpperCase() === upper);
      return {
        tableAlias: tableUpper,
        tableName: table.originalName || tableUpper,
        column: col2,
        columnIndex: colIndex,
        depth
      };
    }
    let found = null;
    let foundDepth = 0;
    for (const [alias, tableInfo] of this.tables) {
      const colIndex = tableInfo.columns.findIndex((c) => c.name.toUpperCase() === upper);
      if (colIndex >= 0) {
        if (found) {
          throw new Error(`Ambiguous column reference: ${name}`);
        }
        found = {
          tableAlias: alias,
          tableName: tableInfo.originalName || alias,
          column: tableInfo.columns[colIndex],
          columnIndex: colIndex,
          depth: 0
        };
      }
    }
    if (found) return found;
    if (this.parent) {
      const parentResult = this.parent.resolveColumn(name);
      if (parentResult) {
        return { ...parentResult, depth: parentResult.depth + 1 };
      }
    }
    return null;
  }
  getAllColumns() {
    const result = [];
    for (const [alias, tableInfo] of this.tables) {
      for (let i = 0; i < tableInfo.columns.length; i++) {
        result.push({
          tableAlias: alias,
          tableName: tableInfo.originalName || alias,
          column: tableInfo.columns[i],
          columnIndex: i
        });
      }
    }
    return result;
  }
  getTableColumns(tableAlias) {
    const upper = tableAlias.toUpperCase();
    const tableInfo = this.tables.get(upper);
    if (!tableInfo) return null;
    return tableInfo.columns.map((col2, i) => ({
      tableAlias: upper,
      tableName: tableInfo.originalName || upper,
      column: col2,
      columnIndex: i
    }));
  }
  child() {
    return new _BinderScope(this);
  }
};

// src/binder/binder.js
init_expression_binder();
var Binder = class {
  constructor(catalog, functionRegistry) {
    this.catalog = catalog;
    this.functionRegistry = functionRegistry;
    this.cteScopes = /* @__PURE__ */ new Map();
    this.aggregatesFound = [];
  }
  bind(ast) {
    const scope = new BinderScope();
    return this.bindQuery(ast, scope);
  }
  bindQuery(node, scope) {
    if (node.kind === NodeKind.SET_OP) {
      return this.bindSetOp(node, scope);
    }
    return this.bindSelect(node, scope);
  }
  bindSetOp(node, scope) {
    const left = this.bindQuery(node.left, scope);
    const right = this.bindQuery(node.right, scope);
    return {
      type: "SetOp",
      op: node.op,
      all: node.all,
      left,
      right,
      outputColumns: left.outputColumns
    };
  }
  bindSelect(node, scope) {
    if (node.withClause) {
      this.bindWithClause(node.withClause, scope);
    }
    const fromScope = scope.child();
    let plan = null;
    if (node.from) {
      plan = this.bindFrom(node.from, fromScope);
    }
    const savedAggregates = this.aggregatesFound;
    this.aggregatesFound = [];
    let outputColumns = [];
    const boundSelectItems = this.bindSelectItems(node.selectItems, fromScope);
    let where = null;
    if (node.where) {
      where = this.bindExpression(node.where, fromScope);
    }
    const aggregates = [...this.aggregatesFound];
    const selectAliasMap = /* @__PURE__ */ new Map();
    for (const item of boundSelectItems) {
      const alias = item.alias || item.inferredName;
      if (alias) {
        selectAliasMap.set(alias.toUpperCase(), item.expr);
      }
    }
    let groupBy = null;
    if (node.groupBy) {
      groupBy = node.groupBy.map((expr2) => {
        if (expr2.kind === NodeKind.COLUMN_REF && !expr2.table) {
          const aliasExpr = selectAliasMap.get(expr2.name.toUpperCase());
          if (aliasExpr) return aliasExpr;
        }
        return this.bindExpression(expr2, fromScope);
      });
    }
    let having = null;
    if (node.having) {
      having = this.bindExpression(node.having, fromScope);
      aggregates.push(...this.aggregatesFound.slice(aggregates.length));
    }
    this.aggregatesFound = savedAggregates;
    let orderBy = null;
    if (node.orderBy) {
      orderBy = node.orderBy.map((ok) => {
        if (ok.expr.kind === "ColumnRef" && !ok.expr.table) {
          const aliasExpr = selectAliasMap.get(ok.expr.name.toUpperCase());
          if (aliasExpr) {
            return { expr: aliasExpr, direction: ok.direction, nullOrder: ok.nullOrder };
          }
        }
        return {
          expr: this.bindExpression(ok.expr, fromScope),
          direction: ok.direction,
          nullOrder: ok.nullOrder
        };
      });
    }
    let limit = null;
    if (node.limit) {
      limit = this.bindExpression(node.limit, fromScope);
    }
    let offset = null;
    if (node.offset) {
      offset = this.bindExpression(node.offset, fromScope);
    }
    outputColumns = boundSelectItems.map((item, i) => ({
      name: item.alias || item.inferredName || `col${i}`,
      expr: item.expr,
      dataType: getExprType(item.expr)
    }));
    return {
      type: "BoundSelect",
      plan,
      selectItems: boundSelectItems,
      where,
      groupBy,
      aggregates,
      having,
      orderBy,
      limit,
      offset,
      distinct: node.distinct,
      outputColumns
    };
  }
  bindWithClause(withClause, scope) {
    for (const cte of withClause.ctes) {
      const cteScope = scope.child();
      const bound = this.bindQuery(cte.query, cteScope);
      const columns = bound.outputColumns.map((col2, i) => ({
        name: cte.columnAliases ? cte.columnAliases[i] : col2.name,
        dataType: col2.dataType
      }));
      this.cteScopes.set(cte.name.toUpperCase(), {
        name: cte.name,
        columns,
        bound
      });
    }
  }
  bindFrom(node, scope) {
    switch (node.kind) {
      case NodeKind.TABLE_REF:
        return this.bindTableRef(node, scope);
      case NodeKind.JOIN_REF:
        return this.bindJoinRef(node, scope);
      case NodeKind.SUBQUERY_REF:
        return this.bindSubqueryRef(node, scope);
      default:
        throw new Error(`Unknown FROM node kind: ${node.kind}`);
    }
  }
  bindTableRef(node, scope) {
    const upperName = node.name.toUpperCase();
    const cte = this.cteScopes.get(upperName);
    if (cte) {
      scope.addTable(node.alias, {
        originalName: cte.name,
        columns: cte.columns,
        isCTE: true
      });
      return {
        type: "CTERef",
        cteName: cte.name,
        alias: node.alias.toUpperCase(),
        columns: cte.columns,
        query: cte.bound
      };
    }
    const tableInfo = this.catalog.getTable(node.name);
    if (!tableInfo) {
      throw new Error(`Unknown table: ${node.name}`);
    }
    scope.addTable(node.alias, {
      originalName: tableInfo.name,
      columns: tableInfo.columns
    });
    return {
      type: "TableRef",
      tableName: tableInfo.name,
      alias: node.alias.toUpperCase(),
      columns: tableInfo.columns
    };
  }
  bindJoinRef(node, scope) {
    const left = this.bindFrom(node.left, scope);
    const right = this.bindFrom(node.right, scope);
    let condition = null;
    if (node.natural) {
      condition = this.buildNaturalJoinCondition(left, right, scope);
    } else if (node.usingColumns) {
      condition = this.buildUsingCondition(node.usingColumns, left, right, scope);
    } else if (node.condition) {
      condition = this.bindExpression(node.condition, scope);
    }
    return {
      type: "JoinRef",
      joinType: node.joinType,
      left,
      right,
      condition
    };
  }
  buildNaturalJoinCondition(left, right, scope) {
    const leftCols = this.getRefColumnNames(left);
    const rightCols = this.getRefColumnNames(right);
    const common = leftCols.filter((n) => rightCols.includes(n));
    if (common.length === 0) return null;
    return this.buildUsingCondition(common, left, right, scope);
  }
  buildUsingCondition(columnNames, left, right, scope) {
    let condition = null;
    for (const colName of columnNames) {
      const leftRef = scope.resolveColumn(colName, this.getRefAlias(left));
      const rightRef = scope.resolveColumn(colName, this.getRefAlias(right));
      if (!leftRef || !rightRef) throw new Error(`USING column not found: ${colName}`);
      const eq = BoundBinary(
        "=",
        BoundColumnRef(leftRef.tableAlias, leftRef.column.name, leftRef.columnIndex, leftRef.column.dataType),
        BoundColumnRef(rightRef.tableAlias, rightRef.column.name, rightRef.columnIndex, rightRef.column.dataType),
        DataType.BOOLEAN
      );
      condition = condition ? BoundBinary("AND", condition, eq, DataType.BOOLEAN) : eq;
    }
    return condition;
  }
  getRefColumnNames(ref) {
    const cols = ref.columns || [];
    return cols.map((c) => c.name.toUpperCase());
  }
  getRefAlias(ref) {
    return ref.alias || ref.tableName || ref.cteName || null;
  }
  bindSubqueryRef(node, scope) {
    const subScope = scope.child();
    const bound = this.bindQuery(node.query, subScope);
    const alias = node.alias || "_subquery";
    scope.addTable(alias, {
      originalName: alias,
      columns: bound.outputColumns.map((c) => ({ name: c.name, dataType: c.dataType }))
    });
    return {
      type: "SubqueryRef",
      alias: alias.toUpperCase(),
      query: bound,
      columns: bound.outputColumns
    };
  }
  bindSelectItems(items, scope) {
    const result = [];
    for (const item of items) {
      if (item.expr.kind === NodeKind.ALL_COLUMNS) {
        const expanded = this.expandStar(item.expr, scope);
        result.push(...expanded);
      } else {
        const expr2 = this.bindExpression(item.expr, scope);
        result.push({
          expr: expr2,
          alias: item.alias,
          inferredName: this.inferColumnName(item.expr)
        });
      }
    }
    return result;
  }
  expandStar(node, scope) {
    if (node.table) {
      const cols = scope.getTableColumns(node.table);
      if (!cols) throw new Error(`Unknown table for star: ${node.table}`);
      return cols.map((c) => ({
        expr: BoundColumnRef(c.tableAlias, c.column.name, c.columnIndex, c.column.dataType),
        alias: null,
        inferredName: c.column.name
      }));
    }
    return scope.getAllColumns().map((c) => ({
      expr: BoundColumnRef(c.tableAlias, c.column.name, c.columnIndex, c.column.dataType),
      alias: null,
      inferredName: c.column.name
    }));
  }
  bindExpression(node, scope) {
    if (!node) return null;
    switch (node.kind) {
      case NodeKind.COLUMN_REF:
        return this.bindColumnRef(node, scope);
      case NodeKind.LITERAL:
        return this.bindLiteral(node);
      case NodeKind.BINARY_EXPR:
        return this.bindBinaryExpr(node, scope);
      case NodeKind.UNARY_EXPR:
        return this.bindUnaryExpr(node, scope);
      case NodeKind.AGGREGATE_CALL:
        return this.bindAggregateCall(node, scope);
      case NodeKind.FUNCTION_CALL:
        return this.bindFunctionCall(node, scope);
      case NodeKind.CASE_EXPR:
        return this.bindCaseExpr(node, scope);
      case NodeKind.CAST_EXPR:
        return this.bindCastExpr(node, scope);
      case NodeKind.BETWEEN_EXPR:
        return BoundBetween(
          this.bindExpression(node.expr, scope),
          this.bindExpression(node.low, scope),
          this.bindExpression(node.high, scope),
          node.negated
        );
      case NodeKind.IN_EXPR:
        return this.bindInExpr(node, scope);
      case NodeKind.LIKE_EXPR:
        return BoundLike(
          this.bindExpression(node.expr, scope),
          this.bindExpression(node.pattern, scope),
          node.negated
        );
      case NodeKind.IS_NULL_EXPR:
        return BoundIsNull(this.bindExpression(node.expr, scope), node.negated);
      case NodeKind.EXISTS_EXPR:
        return this.bindExistsExpr(node, scope);
      case NodeKind.SUBQUERY_EXPR:
        return this.bindSubqueryExpr(node, scope);
      case NodeKind.EXTRACT_EXPR:
        return BoundExtract(node.field, this.bindExpression(node.source, scope));
      case NodeKind.SUBSTRING_EXPR:
        return BoundFunction("SUBSTRING", [
          this.bindExpression(node.expr, scope),
          this.bindExpression(node.from, scope),
          node.length ? this.bindExpression(node.length, scope) : null
        ].filter(Boolean), DataType.VARCHAR);
      case NodeKind.INTERVAL_EXPR:
        return BoundInterval(parseInt(node.value, 10), node.unit);
      case NodeKind.WINDOW_CALL:
        return this.bindWindowCall(node, scope);
      case NodeKind.ALL_COLUMNS:
        throw new Error("Star expression in unexpected position");
      default:
        throw new Error(`Unhandled expression kind: ${node.kind}`);
    }
  }
  bindWindowCall(node, scope) {
    const args = node.args.map((a) => this.bindExpression(a, scope));
    const partitionBy = node.windowSpec.partitionBy.map((e) => this.bindExpression(e, scope));
    const orderBy = node.windowSpec.orderBy.map((ok) => ({
      expr: this.bindExpression(ok.expr, scope),
      direction: ok.direction,
      nullOrder: ok.nullOrder
    }));
    const resultType = this.inferWindowType(node.name, args);
    return BoundWindow(node.name.toUpperCase(), args, partitionBy, orderBy, resultType);
  }
  bindColumnRef(node, scope) {
    const resolved = scope.resolveColumn(node.name, node.table);
    if (!resolved) {
      throw new Error(`Unknown column: ${node.table ? `${node.table}.` : ""}${node.name}`);
    }
    return BoundColumnRef(
      resolved.tableAlias,
      resolved.column.name,
      resolved.columnIndex,
      resolved.column.dataType,
      resolved.depth
    );
  }
  bindLiteral(node) {
    if (node.value === null) {
      return BoundLiteral(null, null);
    }
    if (node.dataType === "DATE") {
      const [y, m, d] = node.value.split("-").map(Number);
      return BoundLiteral(dateToEpochDays(y, m, d), DataType.DATE);
    }
    if (node.dataType === "TIMESTAMP") {
      const parts = node.value.split(/[T ]/);
      const [y, mo, d] = parts[0].split("-").map(Number);
      let h = 0, mi = 0, s = 0, ms = 0;
      if (parts[1]) {
        const timeParts = parts[1].split(":");
        h = Number(timeParts[0]) || 0;
        mi = Number(timeParts[1]) || 0;
        if (timeParts[2]) {
          const secParts = timeParts[2].split(".");
          s = Number(secParts[0]) || 0;
          ms = secParts[1] ? Number(secParts[1].padEnd(3, "0").slice(0, 3)) : 0;
        }
      }
      return BoundLiteral(timestampToEpochMs(y, mo, d, h, mi, s, ms), DataType.TIMESTAMP);
    }
    if (node.dataType === "BOOLEAN") {
      return BoundLiteral(node.value, DataType.BOOLEAN);
    }
    if (node.dataType === "VARCHAR") {
      return BoundLiteral(node.value, DataType.VARCHAR);
    }
    if (typeof node.value === "number") {
      if (Number.isInteger(node.value)) {
        return BoundLiteral(node.value, DataType.INT32);
      }
      return BoundLiteral(node.value, DataType.FLOAT64);
    }
    return BoundLiteral(node.value, DataType.VARCHAR);
  }
  bindBinaryExpr(node, scope) {
    const left = this.bindExpression(node.left, scope);
    const right = this.bindExpression(node.right, scope);
    const op = node.op;
    if (["=", "<>", "<", ">", "<=", ">="].includes(op)) {
      return BoundBinary(op, left, right, DataType.BOOLEAN);
    }
    if (["AND", "OR"].includes(op)) {
      return BoundBinary(op, left, right, DataType.BOOLEAN);
    }
    if (["||"].includes(op)) {
      return BoundBinary(op, left, right, DataType.VARCHAR);
    }
    const resultType = this.inferArithmeticType(getExprType(left), getExprType(right));
    return BoundBinary(op, left, right, resultType);
  }
  bindUnaryExpr(node, scope) {
    const operand = this.bindExpression(node.operand, scope);
    if (node.op === "NOT") {
      return BoundUnary("NOT", operand, DataType.BOOLEAN);
    }
    return BoundUnary(node.op, operand, getExprType(operand));
  }
  bindAggregateCall(node, scope) {
    const args = node.args.map((a) => this.bindExpression(a, scope));
    const resultType = this.inferAggregateType(node.name, args);
    const bound = BoundAggregate(node.name, args, node.distinct, resultType);
    this.aggregatesFound.push(bound);
    return bound;
  }
  bindFunctionCall(node, scope) {
    const args = node.args.map((a) => this.bindExpression(a, scope));
    const resultType = this.inferFunctionType(node.name, args);
    return BoundFunction(node.name.toUpperCase(), args, resultType);
  }
  bindCaseExpr(node, scope) {
    const operand = node.operand ? this.bindExpression(node.operand, scope) : null;
    const whenClauses = node.whenClauses.map((wc) => ({
      condition: this.bindExpression(wc.condition, scope),
      result: this.bindExpression(wc.result, scope)
    }));
    const elseExpr = node.elseExpr ? this.bindExpression(node.elseExpr, scope) : null;
    const resultType = getExprType(whenClauses[0]?.result) || DataType.VARCHAR;
    return BoundCase(operand, whenClauses, elseExpr, resultType);
  }
  bindCastExpr(node, scope) {
    const expr2 = this.bindExpression(node.expr, scope);
    const targetType = this.resolveTypeName(node.targetType);
    return BoundCast(expr2, targetType);
  }
  bindInExpr(node, scope) {
    const expr2 = this.bindExpression(node.expr, scope);
    if (node.list.kind === NodeKind.SUBQUERY_EXPR) {
      const subScope = scope.child();
      const subPlan = this.bindQuery(node.list.query, subScope);
      return BoundInList(expr2, BoundSubquery(subPlan, "IN"), node.negated);
    }
    const list = node.list.map((e) => this.bindExpression(e, scope));
    return BoundInList(expr2, list, node.negated);
  }
  bindExistsExpr(node, scope) {
    const subScope = scope.child();
    const subPlan = this.bindQuery(node.query, subScope);
    return BoundExists(subPlan, node.negated);
  }
  bindSubqueryExpr(node, scope) {
    const subScope = scope.child();
    const subPlan = this.bindQuery(node.query, subScope);
    return BoundSubquery(subPlan, "SCALAR");
  }
  inferColumnName(node) {
    if (node.kind === NodeKind.COLUMN_REF) return node.name;
    if (node.kind === NodeKind.AGGREGATE_CALL) return node.name.toLowerCase();
    if (node.kind === NodeKind.FUNCTION_CALL) return node.name.toLowerCase();
    return null;
  }
  inferArithmeticType(left, right) {
    if (left === DataType.FLOAT64 || right === DataType.FLOAT64) return DataType.FLOAT64;
    if (left === DataType.DECIMAL || right === DataType.DECIMAL) return DataType.DECIMAL;
    if (left === DataType.INT64 || right === DataType.INT64) return DataType.INT64;
    if (left === DataType.DATE) return DataType.DATE;
    return DataType.INT32;
  }
  inferAggregateType(name, args) {
    switch (name.toUpperCase()) {
      case "COUNT":
      case "COUNT_STAR":
        return DataType.INT64;
      case "SUM":
        return args[0] ? getExprType(args[0]) || DataType.FLOAT64 : DataType.FLOAT64;
      case "AVG":
        return DataType.FLOAT64;
      case "MIN":
      case "MAX":
        return args[0] ? getExprType(args[0]) || DataType.FLOAT64 : DataType.FLOAT64;
      default:
        return DataType.FLOAT64;
    }
  }
  inferFunctionType(name, args) {
    switch (name.toUpperCase()) {
      case "SUBSTRING":
      case "TRIM":
      case "UPPER":
      case "LOWER":
      case "REPLACE":
        return DataType.VARCHAR;
      case "EXTRACT":
        return DataType.INT32;
      case "LENGTH":
        return DataType.INT32;
      case "ABS":
      case "ROUND":
      case "SQRT":
        return args[0] ? getExprType(args[0]) : DataType.FLOAT64;
      case "COALESCE":
      case "NULLIF":
        return args[0] ? getExprType(args[0]) : null;
      default:
        return DataType.VARCHAR;
    }
  }
  inferWindowType(name, args) {
    switch (name.toUpperCase()) {
      case "ROW_NUMBER":
      case "RANK":
      case "DENSE_RANK":
        return DataType.INT64;
      case "LAG":
      case "LEAD":
        return args[0] ? getExprType(args[0]) : DataType.VARCHAR;
      case "SUM":
      case "AVG":
      case "MIN":
      case "MAX":
      case "COUNT":
      case "COUNT_STAR":
        return this.inferAggregateType(name, args);
      default:
        return DataType.FLOAT64;
    }
  }
  resolveTypeName(typeName) {
    const name = typeName.name.toUpperCase();
    const map = {
      "INTEGER": DataType.INT32,
      "INT": DataType.INT32,
      "INT32": DataType.INT32,
      "BIGINT": DataType.INT64,
      "INT64": DataType.INT64,
      "FLOAT": DataType.FLOAT64,
      "DOUBLE": DataType.FLOAT64,
      "REAL": DataType.FLOAT64,
      "DECIMAL": DataType.DECIMAL,
      "NUMERIC": DataType.DECIMAL,
      "VARCHAR": DataType.VARCHAR,
      "TEXT": DataType.VARCHAR,
      "CHAR": DataType.VARCHAR,
      "DATE": DataType.DATE,
      "TIMESTAMP": DataType.TIMESTAMP,
      "DATETIME": DataType.TIMESTAMP,
      "BOOLEAN": DataType.BOOLEAN,
      "BOOL": DataType.BOOLEAN
    };
    return map[name] || DataType.VARCHAR;
  }
};

// src/planner/logical-planner.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var _cteIdCounter = 0;
function projectionExpr(item) {
  const name = item.alias || item.inferredName;
  if (!name) return item.expr;
  return { ...item.expr, outputName: name };
}
var LogicalPlanner = class {
  constructor() {
    this.cteMap = /* @__PURE__ */ new Map();
  }
  plan(boundQuery) {
    return this.planQuery(boundQuery);
  }
  planQuery(bound) {
    if (bound.type === "SetOp") {
      return this.planSetOp(bound);
    }
    return this.planSelect(bound);
  }
  planSetOp(bound) {
    const left = this.planQuery(bound.left);
    const right = this.planQuery(bound.right);
    return LogicalUnion(left, right, bound.all);
  }
  planSelect(bound) {
    let node = null;
    if (bound.plan) {
      node = this.planFrom(bound.plan);
    } else {
      node = LogicalSingleRow();
    }
    if (bound.where) {
      const { expr: expr2, subqueryJoins } = this.extractSubqueries(bound.where, node);
      for (const sj of subqueryJoins) {
        node = sj(node);
      }
      if (expr2) {
        node = LogicalFilter(expr2, node);
      }
    }
    if (bound.aggregates.length > 0 || bound.groupBy) {
      node = LogicalAggregate(
        bound.groupBy || [],
        bound.aggregates,
        node
      );
    }
    if (bound.having) {
      const { expr: expr2, subqueryJoins } = this.extractSubqueries(bound.having, node);
      for (const sj of subqueryJoins) {
        node = sj(node);
      }
      if (expr2) {
        node = LogicalFilter(expr2, node);
      }
    }
    for (let i = 0; i < bound.selectItems.length; i++) {
      const { expr: expr2, subqueryJoins } = this.extractSubqueries(bound.selectItems[i].expr, node);
      for (const sj of subqueryJoins) {
        node = sj(node);
      }
      bound.selectItems[i] = { ...bound.selectItems[i], expr: expr2 };
    }
    const windowExprs = [];
    for (const item of bound.selectItems) {
      this._collectWindows(item.expr, windowExprs);
    }
    if (windowExprs.length > 0) {
      node = LogicalWindow(windowExprs, node);
    }
    if (bound.distinct) {
      const projections = bound.selectItems.map(projectionExpr);
      node = LogicalProject(projections, node);
      node = LogicalDistinct(node);
      if (bound.orderBy) {
        node = LogicalSort(bound.orderBy, node);
      }
    } else {
      if (bound.orderBy) {
        node = LogicalSort(bound.orderBy, node);
      }
      const projections = bound.selectItems.map(projectionExpr);
      node = LogicalProject(projections, node);
    }
    if (bound.limit) {
      const limitVal = bound.limit.value;
      const offsetVal = bound.offset ? bound.offset.value : 0;
      node = LogicalLimit(limitVal, offsetVal, node);
    }
    return node;
  }
  planFrom(bound) {
    switch (bound.type) {
      case "TableRef":
        return LogicalScan(bound.tableName, bound.columns, bound.alias);
      case "CTERef": {
        const cteId = _cteIdCounter++;
        const ctePlan = bound.query.prebuiltPlan ?? this.planQuery(bound.query);
        this.cteMap.set(bound.cteName.toUpperCase(), ctePlan);
        return LogicalCTEScan(bound.cteName, cteId);
      }
      case "JoinRef": {
        const left = this.planFrom(bound.left);
        const right = this.planFrom(bound.right);
        const joinType = bound.joinType === "CROSS" ? JoinType.CROSS : JoinType[bound.joinType];
        return LogicalJoin(joinType, bound.condition, left, right);
      }
      case "SubqueryRef": {
        const subPlan = this.planQuery(bound.query);
        return subPlan;
      }
      default:
        throw new Error(`Unknown from type: ${bound.type}`);
    }
  }
  extractSubqueries(expr2, currentPlan) {
    const subqueryJoins = [];
    const transformed = this.walkAndReplace(expr2, (node) => {
      if (node.kind === BoundExprKind.UNARY && node.op === "NOT" && node.operand?.kind === BoundExprKind.EXISTS) {
        const subPlan = this.planQuery(node.operand.plan);
        const correlated = this.findCorrelatedRefs(node.operand.plan);
        subqueryJoins.push(
          (child) => LogicalDependentJoin(child, subPlan, correlated, "NOT_EXISTS", null)
        );
        return null;
      }
      if (node.kind === BoundExprKind.EXISTS) {
        const subPlan = this.planQuery(node.plan);
        const correlated = this.findCorrelatedRefs(node.plan);
        const subqueryType = node.negated ? "NOT_EXISTS" : "EXISTS";
        subqueryJoins.push(
          (child) => LogicalDependentJoin(child, subPlan, correlated, subqueryType, null)
        );
        return null;
      }
      if (node.kind === BoundExprKind.IN_LIST && node.list?.kind === BoundExprKind.SUBQUERY) {
        const subPlan = this.planQuery(node.list.plan);
        const correlated = this.findCorrelatedRefs(node.list.plan);
        const subqueryType = node.negated ? "NOT_IN" : "IN";
        subqueryJoins.push(
          (child) => LogicalDependentJoin(child, subPlan, correlated, subqueryType, node.expr)
        );
        return null;
      }
      if (node.kind === BoundExprKind.SUBQUERY && node.subqueryType === "SCALAR") {
        const subPlan = this.planQuery(node.plan);
        const correlated = this.findCorrelatedRefs(node.plan);
        subqueryJoins.push(
          (child) => LogicalDependentJoin(child, subPlan, correlated, "SCALAR", null)
        );
        return {
          kind: BoundExprKind.COLUMN_REF,
          tableAlias: "",
          columnName: "_scalar",
          columnIndex: -1,
          dataType: "FLOAT64",
          depth: 0,
          isCorrelated: false
        };
      }
      return node;
    });
    return { expr: transformed, subqueryJoins };
  }
  walkAndReplace(expr2, fn) {
    if (!expr2) return null;
    const result = fn(expr2);
    if (result !== expr2) return result;
    switch (expr2.kind) {
      case BoundExprKind.BINARY:
        return {
          ...expr2,
          left: this.walkAndReplace(expr2.left, fn),
          right: this.walkAndReplace(expr2.right, fn)
        };
      case BoundExprKind.UNARY:
        return { ...expr2, operand: this.walkAndReplace(expr2.operand, fn) };
      case BoundExprKind.CASE:
        return {
          ...expr2,
          operand: expr2.operand ? this.walkAndReplace(expr2.operand, fn) : null,
          whenClauses: expr2.whenClauses.map((wc) => ({
            condition: this.walkAndReplace(wc.condition, fn),
            result: this.walkAndReplace(wc.result, fn)
          })),
          elseExpr: expr2.elseExpr ? this.walkAndReplace(expr2.elseExpr, fn) : null
        };
      case BoundExprKind.BETWEEN:
        return {
          ...expr2,
          expr: this.walkAndReplace(expr2.expr, fn),
          low: this.walkAndReplace(expr2.low, fn),
          high: this.walkAndReplace(expr2.high, fn)
        };
      default:
        return expr2;
    }
  }
  _collectWindows(expr2, out) {
    if (!expr2) return;
    if (expr2.kind === BoundExprKind.WINDOW) {
      out.push(expr2);
      return;
    }
    if (expr2.left) this._collectWindows(expr2.left, out);
    if (expr2.right) this._collectWindows(expr2.right, out);
    if (expr2.operand) this._collectWindows(expr2.operand, out);
    if (expr2.args) for (const a of expr2.args) this._collectWindows(a, out);
    if (expr2.whenClauses) for (const wc of expr2.whenClauses) {
      this._collectWindows(wc.condition, out);
      this._collectWindows(wc.result, out);
    }
    if (expr2.elseExpr) this._collectWindows(expr2.elseExpr, out);
  }
  findCorrelatedRefs(boundQuery) {
    const refs = [];
    this._scanForCorrelated(boundQuery, refs);
    return refs;
  }
  _scanForCorrelated(obj, refs) {
    if (!obj || typeof obj !== "object") return;
    if (obj.kind === BoundExprKind.COLUMN_REF && obj.isCorrelated) {
      refs.push(obj);
      return;
    }
    for (const val of Object.values(obj)) {
      if (Array.isArray(val)) {
        for (const item of val) this._scanForCorrelated(item, refs);
      } else if (val && typeof val === "object") {
        this._scanForCorrelated(val, refs);
      }
    }
  }
};
function createLogicalPlan(boundQuery) {
  const planner = new LogicalPlanner();
  const plan = planner.plan(boundQuery);
  plan._cteMap = planner.cteMap;
  return plan;
}

// src/catalog/function-registry.js
init_buffer_shim();
var FunctionType = {
  SCALAR: "SCALAR",
  AGGREGATE: "AGGREGATE"
};
var FunctionRegistry = class {
  constructor() {
    this.functions = /* @__PURE__ */ new Map();
    this._registerBuiltins();
  }
  register(name, definition) {
    this.functions.set(name.toUpperCase(), definition);
  }
  lookup(name) {
    return this.functions.get(name.toUpperCase()) || null;
  }
  has(name) {
    return this.functions.has(name.toUpperCase());
  }
  isAggregate(name) {
    const fn = this.lookup(name);
    return fn?.type === FunctionType.AGGREGATE;
  }
  isScalar(name) {
    const fn = this.lookup(name);
    return fn?.type === FunctionType.SCALAR;
  }
  _registerBuiltins() {
    const agg = (name, minArgs = 1, maxArgs = 1) => ({
      name: name.toUpperCase(),
      type: FunctionType.AGGREGATE,
      minArgs,
      maxArgs
    });
    const scalar = (name, minArgs, maxArgs) => ({
      name: name.toUpperCase(),
      type: FunctionType.SCALAR,
      minArgs,
      maxArgs: maxArgs ?? minArgs
    });
    this.register("SUM", agg("SUM"));
    this.register("AVG", agg("AVG"));
    this.register("COUNT", agg("COUNT", 0, 1));
    this.register("MIN", agg("MIN"));
    this.register("MAX", agg("MAX"));
    this.register("COUNT_STAR", agg("COUNT_STAR", 0, 0));
    this.register("SUBSTRING", scalar("SUBSTRING", 2, 3));
    this.register("EXTRACT", scalar("EXTRACT", 2, 2));
    this.register("TRIM", scalar("TRIM", 1, 1));
    this.register("UPPER", scalar("UPPER", 1, 1));
    this.register("LOWER", scalar("LOWER", 1, 1));
    this.register("CAST", scalar("CAST", 2, 2));
    this.register("COALESCE", scalar("COALESCE", 1, Infinity));
    this.register("NULLIF", scalar("NULLIF", 2, 2));
    this.register("ABS", scalar("ABS", 1, 1));
    this.register("ROUND", scalar("ROUND", 1, 2));
    this.register("SQRT", scalar("SQRT", 1, 1));
    this.register("LENGTH", scalar("LENGTH", 1, 1));
    this.register("REPLACE", scalar("REPLACE", 3, 3));
  }
};
var defaultFunctionRegistry = new FunctionRegistry();

// src/engine/query-engine.js
init_ast();
init_data_type();
init_column();

// src/storage/table.js
init_buffer_shim();
init_column();
init_chunk();

// src/storage/buffer-pool.js
init_buffer_shim();

// src/utils/lru-cache.js
init_buffer_shim();
var LRUCache = class {
  constructor(maxSize) {
    this._maxSize = maxSize;
    this._map = /* @__PURE__ */ new Map();
    this._head = null;
    this._tail = null;
  }
  _detach(node) {
    if (node.prev) {
      node.prev.next = node.next;
    } else {
      this._head = node.next;
    }
    if (node.next) {
      node.next.prev = node.prev;
    } else {
      this._tail = node.prev;
    }
    node.prev = null;
    node.next = null;
  }
  _prepend(node) {
    node.next = this._head;
    node.prev = null;
    if (this._head) {
      this._head.prev = node;
    }
    this._head = node;
    if (!this._tail) {
      this._tail = node;
    }
  }
  _moveToHead(node) {
    if (node === this._head) return;
    this._detach(node);
    this._prepend(node);
  }
  _evictTail() {
    const evicted = this._tail;
    if (!evicted) return null;
    this._map.delete(evicted.key);
    this._detach(evicted);
    return evicted.key;
  }
  get(key) {
    const node = this._map.get(key);
    if (!node) return void 0;
    this._moveToHead(node);
    return node.value;
  }
  set(key, value) {
    const existing = this._map.get(key);
    if (existing) {
      existing.value = value;
      this._moveToHead(existing);
      return null;
    }
    const node = { key, value, prev: null, next: null };
    this._map.set(key, node);
    this._prepend(node);
    if (this._map.size > this._maxSize) {
      return this._evictTail();
    }
    return null;
  }
  has(key) {
    return this._map.has(key);
  }
  delete(key) {
    const node = this._map.get(key);
    if (!node) return false;
    this._detach(node);
    this._map.delete(key);
    return true;
  }
  clear() {
    this._map.clear();
    this._head = null;
    this._tail = null;
  }
  get size() {
    return this._map.size;
  }
};

// src/storage/buffer-pool.js
var BufferPoolManager = class {
  constructor(maxPages, pageStore) {
    this.maxPages = maxPages;
    this.cache = new LRUCache(maxPages);
    this.pageStore = pageStore;
  }
  clear() {
    this.cache.clear();
    this.pageStore.clear();
  }
  async writePage(pageId, chunk) {
    await this.pageStore.write(pageId, chunk);
  }
  async readPage(pageId) {
    return this.pageStore.read(pageId);
  }
  async fetchPage(pageId, bypassCache = false) {
    const cached = this.cache.get(pageId);
    if (cached !== void 0) {
      return cached;
    }
    const chunk = await this.readPage(pageId);
    if (bypassCache) {
      return chunk;
    }
    this.cache.set(pageId, chunk);
    return chunk;
  }
};

// src/storage/table.js
init_config();
var Table = class {
  constructor(name, schema, pageStore) {
    this.name = name;
    this.schema = schema;
    this.pageIds = [];
    this._rowCount = 0;
    this.bufferPool = new BufferPoolManager(Config.bufferPoolPages, pageStore);
    this.activeChunk = null;
    this.indexes = [];
  }
  getSchema() {
    return this.schema;
  }
  getColumnIndex(columnName) {
    const upper = columnName.toUpperCase();
    return this.schema.findIndex((col2) => col2.name.toUpperCase() === upper);
  }
  getColumn(columnName) {
    return this.schema.find((col2) => col2.name.toUpperCase() === columnName.toUpperCase());
  }
  rowCount() {
    return this._rowCount + (this.activeChunk ? this.activeChunk.size : 0);
  }
  registerIndex(columnIndex, btree) {
    this.indexes.push({ columnIndex, btree });
  }
  async addChunk(chunk) {
    const pageId = `${this.name}_page_${this.pageIds.length}`;
    this.pageIds.push(pageId);
    this._rowCount += chunk.size;
    await this.bufferPool.writePage(pageId, chunk);
    for (const idx of this.indexes) {
      for (let r = 0; r < chunk.size; r++) {
        const key = chunk.columns[idx.columnIndex].get(r);
        if (key !== null && key !== void 0) {
          idx.btree.insert(key, { pageId, rowIndex: r });
        }
      }
    }
  }
  async insertRows(rows) {
    if (!this.activeChunk) {
      this.activeChunk = this._createChunk();
    }
    for (const row of rows) {
      if (this.activeChunk.size >= DEFAULT_CHUNK_SIZE) {
        await this.addChunk(this.activeChunk);
        this.activeChunk = this._createChunk();
      }
      this.activeChunk.appendRow(row);
    }
  }
  async flush() {
    if (this.activeChunk && this.activeChunk.size > 0) {
      await this.addChunk(this.activeChunk);
      this.activeChunk = null;
    }
  }
  async *scan() {
    await this.flush();
    for (const pageId of this.pageIds) {
      const chunk = await this.bufferPool.fetchPage(pageId, true);
      yield chunk;
    }
  }
  async scanAll() {
    await this.flush();
    const chunks = [];
    for (const pageId of this.pageIds) {
      chunks.push(await this.bufferPool.fetchPage(pageId, false));
    }
    return chunks;
  }
  getStatistics() {
    return null;
  }
  _createChunk() {
    return DataChunk.fromSchema(this.schema, DEFAULT_CHUNK_SIZE);
  }
};

// src/optimizer/optimizer.js
init_buffer_shim();
var Optimizer = class {
  constructor() {
    this.passes = [];
  }
  registerPass(pass) {
    this.passes.push(pass);
    return this;
  }
  removePass(name) {
    this.passes = this.passes.filter((p) => p.name !== name);
    return this;
  }
  insertPassBefore(name, pass) {
    const idx = this.passes.findIndex((p) => p.name === name);
    if (idx === -1) {
      this.passes.push(pass);
    } else {
      this.passes.splice(idx, 0, pass);
    }
    return this;
  }
  insertPassAfter(name, pass) {
    const idx = this.passes.findIndex((p) => p.name === name);
    if (idx === -1) {
      this.passes.push(pass);
    } else {
      this.passes.splice(idx + 1, 0, pass);
    }
    return this;
  }
  optimize(plan, context = {}) {
    let current = plan;
    for (const pass of this.passes) {
      current = pass.apply(current, context);
    }
    return current;
  }
  listPasses() {
    return this.passes.map((p) => p.name);
  }
};

// src/optimizer/passes/subquery-unnesting.js
init_buffer_shim();

// src/optimizer/pass.js
init_buffer_shim();
var OptimizationPass = class {
  get name() {
    throw new Error("Subclass must implement name");
  }
  apply(plan, context) {
    throw new Error("Subclass must implement apply()");
  }
};

// src/optimizer/passes/subquery-unnesting.js
init_logical_plan();

// src/planner/plan-visitor.js
init_buffer_shim();
init_logical_plan();
var PlanRewriter = class {
  rewrite(node) {
    const method = `rewrite${node.type}`;
    if (typeof this[method] === "function") {
      return this[method](node);
    }
    return this.rewriteDefault(node);
  }
  rewriteDefault(node) {
    return this.rewriteChildren(node);
  }
  rewriteChildren(node) {
    const children = getChildren(node);
    if (children.length === 0) return node;
    const newChildren = children.map((child) => this.rewrite(child));
    const changed = newChildren.some((child, i) => child !== children[i]);
    return changed ? setChildren(node, newChildren) : node;
  }
};

// src/optimizer/passes/subquery-unnesting.js
init_expression_binder();

// src/optimizer/passes/predicate-pushdown.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var PredicatePushdown = class extends OptimizationPass {
  get name() {
    return "PredicatePushdown";
  }
  apply(plan) {
    const rewriter = new PushdownRewriter();
    return rewriter.rewrite(plan);
  }
};
var PushdownRewriter = class extends PlanRewriter {
  rewriteJoin(node) {
    const rewritten = this.rewriteChildren(node);
    return pushJoinConditionPredicates(rewritten);
  }
  rewriteFilter(node) {
    const child = this.rewrite(node.children[0]);
    const predicates = splitConjuncts(node.condition);
    return pushPredicates(predicates, child);
  }
  rewriteDefault(node) {
    return this.rewriteChildren(node);
  }
};
function pushJoinConditionPredicates(joinNode) {
  if (!joinNode.condition) return joinNode;
  const rightRefs = collectPlanRefs(joinNode.children[1]);
  const rightPreds = [];
  const joinPreds = [];
  for (const pred of splitConjuncts(joinNode.condition)) {
    const refs = collectTableRefs(pred);
    const rightOnly = refs.length > 0 && refs.every((r) => refBelongsToPlan(r, rightRefs));
    if (joinNode.joinType === JoinType.LEFT) {
      if (rightOnly) rightPreds.push(pred);
      else joinPreds.push(pred);
    } else {
      joinPreds.push(pred);
    }
  }
  if (rightPreds.length === 0) return joinNode;
  let right = joinNode.children[1];
  if (rightPreds.length > 0) right = pushPredicates(rightPreds, right);
  const result = {
    ...LogicalJoin(
      joinNode.joinType === JoinType.CROSS && joinPreds.length > 0 ? JoinType.INNER : joinNode.joinType,
      combineConjuncts(joinPreds),
      joinNode.children[0],
      right
    ),
    ...copyJoinProperties(joinNode)
  };
  return result;
}
function copyJoinProperties(joinNode) {
  const props = {};
  if (joinNode.markColumn) props.markColumn = joinNode.markColumn;
  for (const key of Object.keys(joinNode)) {
    if (key.startsWith("_")) props[key] = joinNode[key];
  }
  return props;
}
function pushPredicates(predicates, target) {
  if (target.type === PlanNodeType.JOIN) {
    return pushIntoJoin(predicates, target);
  }
  if (target.type === PlanNodeType.FILTER) {
    const innerPreds = splitConjuncts(target.condition);
    const allPreds = [...innerPreds, ...predicates];
    const child = target.children[0];
    return pushPredicates(allPreds, child);
  }
  if (target.type === PlanNodeType.AGGREGATE && target.groupBy && target.groupBy.length > 0) {
    const groupByRefs = /* @__PURE__ */ new Set();
    for (const gb of target.groupBy) {
      if (gb.kind === BoundExprKind.COLUMN_REF) {
        groupByRefs.add(`${(gb.tableAlias || "").toUpperCase()}.${(gb.columnName || "").toUpperCase()}`);
      }
    }
    const pushable = [];
    const remaining = [];
    for (const pred of predicates) {
      const refs = collectTableRefs(pred);
      if (refs.length > 0 && !containsAggregate(pred) && refs.every((r) => groupByRefs.has(`${r.tableAlias}.${r.columnName}`))) {
        pushable.push(pred);
      } else {
        remaining.push(pred);
      }
    }
    if (pushable.length > 0) {
      const newChild = pushPredicates(pushable, target.children[0]);
      const newAgg = { ...target, children: [newChild] };
      if (remaining.length === 0) return newAgg;
      return LogicalFilter(combineConjuncts(remaining), newAgg);
    }
  }
  if (target.type === PlanNodeType.PROJECT) {
    const pushable = [];
    const remaining = [];
    for (const pred of predicates) {
      if (canPushThroughProject(pred, target)) {
        pushable.push(pred);
      } else {
        remaining.push(pred);
      }
    }
    if (pushable.length > 0) {
      const newChild = pushPredicates(pushable, target.children[0]);
      const newProject = { ...target, children: [newChild] };
      if (remaining.length === 0) return newProject;
      return LogicalFilter(combineConjuncts(remaining), newProject);
    }
  }
  if (predicates.length === 0) return target;
  return LogicalFilter(combineConjuncts(predicates), target);
}
function canPushThroughProject(pred, projectNode) {
  const predRefs = /* @__PURE__ */ new Set();
  _walkExpr(pred, (e) => {
    if (e.kind === BoundExprKind.COLUMN_REF) {
      predRefs.add({
        tableAlias: (e.tableAlias || "").toUpperCase(),
        columnName: (e.columnName || "").toUpperCase()
      });
    }
  });
  const childRefs = collectPlanRefs(projectNode.children[0]);
  for (const ref of predRefs) {
    if (!refBelongsToPlan(ref, childRefs)) return false;
  }
  return true;
}
function pushIntoJoin(predicates, joinNode) {
  const leftRefs = collectPlanRefs(joinNode.children[0]);
  const rightRefs = collectPlanRefs(joinNode.children[1]);
  const leftPreds = [];
  const rightPreds = [];
  const joinPreds = [];
  const remaining = [];
  for (const pred of predicates) {
    const refs = collectTableRefs(pred);
    const leftOnly = refs.every((r) => refBelongsToPlan(r, leftRefs));
    const rightOnly = refs.every((r) => refBelongsToPlan(r, rightRefs));
    if (joinNode.joinType === JoinType.INNER || joinNode.joinType === JoinType.CROSS) {
      if (leftOnly) leftPreds.push(pred);
      else if (rightOnly) rightPreds.push(pred);
      else joinPreds.push(pred);
    } else if (joinNode.joinType === JoinType.LEFT) {
      if (leftOnly) leftPreds.push(pred);
      else if (rightOnly && rejectsNulls(pred)) {
        rightPreds.push(pred);
        joinNode = { ...joinNode, joinType: JoinType.INNER };
      } else {
        remaining.push(pred);
      }
    } else if (joinNode.joinType === JoinType.SEMI || joinNode.joinType === JoinType.ANTI || joinNode.joinType === JoinType.MARK) {
      if (leftOnly) leftPreds.push(pred);
      else if (rightOnly) rightPreds.push(pred);
      else remaining.push(pred);
    } else {
      remaining.push(pred);
    }
  }
  let left = joinNode.children[0];
  let right = joinNode.children[1];
  if (leftPreds.length > 0) left = pushPredicates(leftPreds, left);
  if (rightPreds.length > 0) right = pushPredicates(rightPreds, right);
  let joinCondition = joinNode.condition;
  if (joinPreds.length > 0) {
    const allJoinPreds = joinCondition ? [joinCondition, ...joinPreds] : joinPreds;
    joinCondition = combineConjuncts(allJoinPreds);
  }
  let result = LogicalJoin(
    joinNode.joinType === JoinType.CROSS && joinCondition ? JoinType.INNER : joinNode.joinType,
    joinCondition,
    left,
    right
  );
  if (joinNode.markColumn) result.markColumn = joinNode.markColumn;
  if (remaining.length > 0) {
    result = LogicalFilter(combineConjuncts(remaining), result);
  }
  return result;
}
function rejectsNulls(pred) {
  if (pred.kind === BoundExprKind.BINARY) {
    return ["=", "<>", "<", ">", "<=", ">="].includes(pred.op);
  }
  if (pred.kind === BoundExprKind.IS_NULL && !pred.negated) {
    return false;
  }
  return true;
}
function splitConjuncts(expr2) {
  if (!expr2) return [];
  if (expr2.kind === BoundExprKind.BINARY && expr2.op === "AND") {
    return [...splitConjuncts(expr2.left), ...splitConjuncts(expr2.right)];
  }
  return [expr2];
}
function combineConjuncts(preds) {
  if (preds.length === 0) return null;
  if (preds.length === 1) return preds[0];
  return preds.reduce((acc, p) => ({
    kind: BoundExprKind.BINARY,
    op: "AND",
    left: acc,
    right: p,
    resultType: "BOOLEAN"
  }));
}
function collectPlanRefs(node) {
  const refs = { aliases: /* @__PURE__ */ new Set(), columns: /* @__PURE__ */ new Set() };
  addOutputRefs(node, refs);
  refs.aliases.delete("");
  refs.columns.delete("");
  return refs;
}
function addOutputRefs(node, refs) {
  if (!node) return;
  if (node.type === PlanNodeType.SCAN) {
    refs.aliases.add(node.alias?.toUpperCase() || node.table?.toUpperCase());
    for (const col2 of node.columns || []) {
      refs.columns.add((col2.name || col2.columnName || "").toUpperCase());
    }
    return;
  }
  if (node.type === PlanNodeType.CTE_SCAN) {
    refs.aliases.add((node.alias || node.cteName || "").toUpperCase());
    return;
  }
  if (node.type === PlanNodeType.PROJECT) {
    for (const expr2 of node.expressions || []) {
      refs.columns.add((expr2.outputName || expr2.alias || expr2.name || expr2.columnName || "").toUpperCase());
    }
    return;
  }
  if (node.type === PlanNodeType.AGGREGATE) {
    for (const expr2 of node.groupBy || []) {
      refs.columns.add((expr2.outputName || expr2.alias || expr2.name || expr2.columnName || "").toUpperCase());
    }
    for (const agg of node.aggregates || []) {
      refs.columns.add((agg.outputName || agg.alias || agg.name || "").toUpperCase());
    }
    return;
  }
  if (node.type === PlanNodeType.JOIN || node.type === PlanNodeType.UNION) {
    for (const child of getChildren(node)) addOutputRefs(child, refs);
    return;
  }
  if (node.children?.[0]) addOutputRefs(node.children[0], refs);
}
function refBelongsToPlan(ref, planRefs) {
  if (ref.tableAlias) return planRefs.aliases.has(ref.tableAlias);
  return planRefs.columns.has(ref.columnName);
}
function containsAggregate(expr2) {
  let found = false;
  _walkExpr(expr2, (e) => {
    if (e.kind === BoundExprKind.AGGREGATE) found = true;
  });
  return found;
}
function collectTableRefs(expr2) {
  const keys = /* @__PURE__ */ new Set();
  _walkExpr(expr2, (e) => {
    if (e.kind === BoundExprKind.COLUMN_REF) {
      keys.add(`${(e.tableAlias || "").toUpperCase()}.${(e.columnName || "").toUpperCase()}`);
    }
  });
  return [...keys].map((key) => {
    const [tableAlias, columnName] = key.split(".");
    return { tableAlias, columnName };
  });
}
function _walkExpr(expr2, fn) {
  if (!expr2 || typeof expr2 !== "object") return;
  fn(expr2);
  if (expr2.left) _walkExpr(expr2.left, fn);
  if (expr2.right) _walkExpr(expr2.right, fn);
  if (expr2.operand) _walkExpr(expr2.operand, fn);
  if (expr2.expr) _walkExpr(expr2.expr, fn);
  if (expr2.low) _walkExpr(expr2.low, fn);
  if (expr2.high) _walkExpr(expr2.high, fn);
  if (expr2.args) for (const a of expr2.args) _walkExpr(a, fn);
  if (expr2.whenClauses) for (const wc of expr2.whenClauses) {
    _walkExpr(wc.condition, fn);
    _walkExpr(wc.result, fn);
  }
  if (expr2.elseExpr) _walkExpr(expr2.elseExpr, fn);
  if (expr2.list && Array.isArray(expr2.list)) for (const item of expr2.list) _walkExpr(item, fn);
  if (expr2.pattern) _walkExpr(expr2.pattern, fn);
  if (expr2.source) _walkExpr(expr2.source, fn);
}

// src/optimizer/passes/subquery-unnesting.js
var SubqueryUnnesting = class extends OptimizationPass {
  get name() {
    return "SubqueryUnnesting";
  }
  apply(plan) {
    let current = plan;
    let changed = true;
    while (changed) {
      const rewriter = new UnnestingRewriter();
      const result = rewriter.rewrite(current);
      changed = rewriter.didChange;
      current = result;
    }
    return current;
  }
};
var UnnestingRewriter = class extends PlanRewriter {
  constructor() {
    super();
    this.didChange = false;
    this.markId = 0;
  }
  rewriteDependentJoin(node) {
    this.didChange = true;
    const left = this.rewrite(node.children[0]);
    const subquery = this.rewrite(node.children[1]);
    const correlated = node.correlatedColumns || [];
    switch (node.subqueryType) {
      case "EXISTS":
        return this.unnestExists(left, subquery, correlated);
      case "NOT_EXISTS":
        return this.unnestNotExists(left, subquery, correlated);
      case "IN":
        return this.unnestIn(left, subquery, correlated, node.condition);
      case "NOT_IN":
        return this.unnestNotIn(left, subquery, correlated, node.condition);
      case "SCALAR":
        return this.unnestScalar(left, subquery, correlated);
      default:
        return setChildren(node, [left, subquery]);
    }
  }
  unnestExists(left, subquery, correlated) {
    const { cleanedPlan, joinCondition } = this.extractCorrelation(subquery, correlated);
    return LogicalJoin(JoinType.SEMI, joinCondition, left, this.removeProjection(cleanedPlan));
  }
  unnestNotExists(left, subquery, correlated) {
    const { cleanedPlan, joinCondition } = this.extractCorrelation(subquery, correlated);
    return LogicalJoin(JoinType.ANTI, joinCondition, left, this.removeProjection(cleanedPlan));
  }
  unnestIn(left, subquery, correlated, inExpr) {
    const { cleanedPlan, joinCondition } = this.extractCorrelation(subquery, correlated);
    const conditions = [];
    if (joinCondition) conditions.push(joinCondition);
    if (inExpr) {
      const outputRef = this.getSubqueryOutputRef(subquery);
      if (outputRef) {
        conditions.push({
          kind: BoundExprKind.BINARY,
          op: "=",
          left: inExpr,
          right: outputRef,
          resultType: "BOOLEAN"
        });
      }
    }
    return LogicalJoin(JoinType.SEMI, combineConjuncts(conditions), left, this.exposeCorrelationColumns(cleanedPlan));
  }
  unnestNotIn(left, subquery, correlated, inExpr) {
    const { cleanedPlan, joinCondition } = this.extractCorrelation(subquery, correlated);
    const conditions = [];
    if (joinCondition) conditions.push(joinCondition);
    if (inExpr) {
      const outputRef = this.getSubqueryOutputRef(subquery);
      if (outputRef) {
        conditions.push({
          kind: BoundExprKind.BINARY,
          op: "=",
          left: inExpr,
          right: outputRef,
          resultType: "BOOLEAN"
        });
      }
    }
    const markName = `__mark_${this.markId++}`;
    const markRef = {
      kind: BoundExprKind.COLUMN_REF,
      tableAlias: "",
      columnName: markName,
      columnIndex: -1,
      dataType: "BOOLEAN",
      depth: 0,
      isCorrelated: false
    };
    const markJoin = {
      ...LogicalJoin(JoinType.MARK, combineConjuncts(conditions), left, this.exposeCorrelationColumns(cleanedPlan)),
      markColumn: markName
    };
    return LogicalFilter({
      kind: BoundExprKind.BINARY,
      op: "=",
      left: markRef,
      right: {
        kind: BoundExprKind.LITERAL,
        value: false,
        dataType: "BOOLEAN"
      },
      resultType: "BOOLEAN"
    }, markJoin);
  }
  unnestScalar(left, subquery, correlated) {
    const { cleanedPlan, joinCondition, correlatedPredicates } = this.extractCorrelation(subquery, correlated);
    const outputRef = this.getSubqueryOutputRef(subquery);
    if (correlated.length > 0 && this.hasAggregate(subquery)) {
      const groupByExprs = this.getInnerCorrelationExprs(correlatedPredicates, correlated);
      const innerPlan = this.removeProjection(cleanedPlan);
      const aggregatedPlan = this.addGroupBy(innerPlan, groupByExprs);
      const scalarPlan = this.projectScalarOutput(aggregatedPlan, groupByExprs, outputRef);
      return LogicalJoin(JoinType.LEFT, joinCondition, left, scalarPlan);
    }
    return LogicalJoin(JoinType.SINGLE, joinCondition, left, this.projectScalarOutput(cleanedPlan, [], outputRef));
  }
  extractCorrelation(subquery, correlated) {
    const correlatedPredicates = [];
    const cleanedPlan = this.removeCorrelatedPredicates(subquery, correlated, correlatedPredicates);
    const joinConditions = correlatedPredicates.map((pred) => {
      return this.rewriteCorrelatedPredicate(pred, correlated);
    });
    return {
      cleanedPlan,
      joinCondition: combineConjuncts(joinConditions),
      correlatedPredicates
    };
  }
  removeCorrelatedPredicates(node, correlated, collected) {
    if (!node) return node;
    if (node.type === PlanNodeType.FILTER) {
      const child = this.removeCorrelatedPredicates(node.children[0], correlated, collected);
      const { correlatedPreds, localPreds } = this.partitionPredicates(node.condition, correlated);
      collected.push(...correlatedPreds);
      if (localPreds.length === 0) return child;
      return LogicalFilter(combineConjuncts(localPreds), child);
    }
    if (node.type === PlanNodeType.PROJECT) {
      const child = this.removeCorrelatedPredicates(node.children[0], correlated, collected);
      return setChildren(node, [child]);
    }
    const children = getChildren(node);
    const newChildren = children.map((c) => this.removeCorrelatedPredicates(c, correlated, collected));
    const changed = newChildren.some((c, i) => c !== children[i]);
    return changed ? setChildren(node, newChildren) : node;
  }
  partitionPredicates(expr2, correlated) {
    const correlatedPreds = [];
    const localPreds = [];
    const preds = this.splitAnd(expr2);
    for (const pred of preds) {
      if (this.hasCorrelatedRef(pred, correlated)) {
        correlatedPreds.push(pred);
      } else {
        localPreds.push(pred);
      }
    }
    return { correlatedPreds, localPreds };
  }
  hasCorrelatedRef(expr2, correlated) {
    if (!expr2 || typeof expr2 !== "object") return false;
    if (expr2.kind === BoundExprKind.COLUMN_REF && expr2.isCorrelated) return true;
    if (expr2.kind === BoundExprKind.COLUMN_REF) {
      return correlated.some(
        (c) => c.tableAlias === expr2.tableAlias && c.columnName === expr2.columnName
      );
    }
    for (const val of Object.values(expr2)) {
      if (Array.isArray(val)) {
        for (const item of val) {
          if (this.hasCorrelatedRef(item, correlated)) return true;
        }
      } else if (val && typeof val === "object") {
        if (this.hasCorrelatedRef(val, correlated)) return true;
      }
    }
    return false;
  }
  rewriteCorrelatedPredicate(pred, correlated) {
    return this.rewriteExprRefs(pred, correlated);
  }
  rewriteExprRefs(expr2, correlated) {
    if (!expr2 || typeof expr2 !== "object") return expr2;
    if (expr2.kind === BoundExprKind.COLUMN_REF && expr2.isCorrelated) {
      return { ...expr2, depth: 0, isCorrelated: false };
    }
    if (expr2.kind === BoundExprKind.COLUMN_REF) {
      const match = correlated.find(
        (c) => c.tableAlias === expr2.tableAlias && c.columnName === expr2.columnName
      );
      if (match) {
        return { ...expr2, depth: 0, isCorrelated: false };
      }
    }
    const result = {};
    for (const [key, val] of Object.entries(expr2)) {
      if (Array.isArray(val)) {
        result[key] = val.map(
          (item) => item && typeof item === "object" ? this.rewriteExprRefs(item, correlated) : item
        );
      } else if (val && typeof val === "object") {
        result[key] = this.rewriteExprRefs(val, correlated);
      } else {
        result[key] = val;
      }
    }
    return result;
  }
  splitAnd(expr2) {
    if (!expr2) return [];
    if (expr2.kind === BoundExprKind.BINARY && expr2.op === "AND") {
      return [...this.splitAnd(expr2.left), ...this.splitAnd(expr2.right)];
    }
    return [expr2];
  }
  getSubqueryOutputRef(subquery) {
    let node = subquery;
    while (node) {
      if (node.type === PlanNodeType.PROJECT && node.expressions?.length > 0) {
        return node.expressions[0];
      }
      node = node.children?.[0];
    }
    return null;
  }
  getInnerCorrelationExprs(correlatedPredicates, correlated) {
    const exprs = [];
    const seen = /* @__PURE__ */ new Set();
    for (const pred of correlatedPredicates) {
      if (pred.kind !== BoundExprKind.BINARY || pred.op !== "=") continue;
      const leftCorrelated = this.hasCorrelatedRef(pred.left, correlated);
      const rightCorrelated = this.hasCorrelatedRef(pred.right, correlated);
      let innerExpr = null;
      if (leftCorrelated && !rightCorrelated) innerExpr = pred.right;
      else if (rightCorrelated && !leftCorrelated) innerExpr = pred.left;
      if (!innerExpr) continue;
      const rewritten = this.rewriteExprRefs(innerExpr, correlated);
      const key = JSON.stringify(rewritten);
      if (!seen.has(key)) {
        seen.add(key);
        exprs.push(rewritten);
      }
    }
    return exprs;
  }
  projectScalarOutput(plan, groupByExprs, outputRef) {
    if (!outputRef) return plan;
    const scalarExpr = { ...outputRef, outputName: "_scalar" };
    return {
      type: PlanNodeType.PROJECT,
      expressions: [...groupByExprs, scalarExpr],
      children: [plan]
    };
  }
  hasAggregate(node) {
    if (!node) return false;
    if (node.type === PlanNodeType.AGGREGATE) return true;
    for (const child of getChildren(node)) {
      if (this.hasAggregate(child)) return true;
    }
    return false;
  }
  removeProjection(node) {
    if (node.type === PlanNodeType.PROJECT) {
      return node.children[0];
    }
    return node;
  }
  exposeCorrelationColumns(node) {
    if (node.type === PlanNodeType.PROJECT && (node.expressions || []).every((e) => e.kind === BoundExprKind.COLUMN_REF)) {
      return node.children[0];
    }
    return node;
  }
  addGroupBy(node, groupByExprs) {
    if (node.type === PlanNodeType.AGGREGATE) {
      return {
        ...node,
        groupBy: [...groupByExprs, ...node.groupBy || []]
      };
    }
    const children = getChildren(node);
    const newChildren = children.map((c) => this.addGroupBy(c, groupByExprs));
    return setChildren(node, newChildren);
  }
};

// src/optimizer/passes/cte-optimization.js
init_buffer_shim();
init_logical_plan();
var CTEOptimization = class extends OptimizationPass {
  get name() {
    return "CTEOptimization";
  }
  apply(plan) {
    const refCounts = countCTERefs(plan);
    const rewriter = new CTERewriter(refCounts);
    return rewriter.rewrite(plan);
  }
};
function countCTERefs(node) {
  const counts = /* @__PURE__ */ new Map();
  _walkPlan(node, (n) => {
    if (n.type === PlanNodeType.CTE_SCAN) {
      const key = n.cteName.toUpperCase();
      counts.set(key, (counts.get(key) || 0) + 1);
    }
  });
  return counts;
}
var CTERewriter = class extends PlanRewriter {
  constructor(refCounts) {
    super();
    this.refCounts = refCounts;
    this.ctePlans = /* @__PURE__ */ new Map();
  }
  rewriteCTEAnchor(node) {
    const producer = this.rewrite(node.children[0]);
    const consumer = this.rewrite(node.children[1]);
    const key = node.cteName.toUpperCase();
    const count2 = this.refCounts.get(key) || 0;
    if (count2 <= 1) {
      this.ctePlans.set(key, producer);
      return consumer;
    }
    this.ctePlans.set(key, LogicalMaterialize(producer));
    return consumer;
  }
  rewriteCTEScan(node) {
    const key = node.cteName.toUpperCase();
    const plan = this.ctePlans.get(key);
    if (plan) {
      return plan;
    }
    return node;
  }
};
function _walkPlan(node, fn) {
  if (!node) return;
  fn(node);
  for (const child of getChildren(node)) _walkPlan(child, fn);
}

// src/optimizer/passes/projection-pushdown.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var ProjectionPushdown = class extends OptimizationPass {
  get name() {
    return "ProjectionPushdown";
  }
  apply(plan) {
    return pruneColumns(plan, null);
  }
};
function pruneColumns(node, required) {
  if (!node) return node;
  switch (node.type) {
    case PlanNodeType.SCAN:
      return pruneScan(node, required);
    case PlanNodeType.PROJECT:
      return pruneProject(node, required);
    case PlanNodeType.FILTER:
      return pruneUnary(node, required ? addExprRefs(required, node.condition) : null);
    case PlanNodeType.JOIN:
      return pruneJoin(node, required);
    case PlanNodeType.AGGREGATE:
      return pruneAggregate(node);
    case PlanNodeType.SORT:
      return pruneSort(node, required);
    case PlanNodeType.LIMIT:
    case PlanNodeType.MATERIALIZE:
      return pruneUnary(node, required);
    case PlanNodeType.DEPENDENT_JOIN:
      return pruneDependentJoin(node, required);
    case PlanNodeType.UNION:
    case PlanNodeType.CTE_ANCHOR:
      return pruneChildren(node, null);
    case PlanNodeType.DISTINCT:
      return pruneUnary(node, null);
    default:
      return pruneChildren(node, required);
  }
}
function pruneScan(node, required) {
  if (!required || required.size === 0) return node;
  const refs = collectPlanRefs2(node);
  const neededCols = node.columns.filter((col2) => refSetNeedsColumn(required, refs.aliases, col2.name));
  if (neededCols.length > 0 && neededCols.length < node.columns.length) {
    return { ...node, columns: neededCols };
  }
  return node;
}
function pruneProject(node, required) {
  const childRequired = /* @__PURE__ */ new Set();
  for (const [index, expr2] of (node.expressions || []).entries()) {
    if (!required || outputNeeded(expr2, required, index)) {
      collectExprColumns(expr2, childRequired);
    }
  }
  const child = pruneColumns(node.children[0], childRequired);
  return child !== node.children[0] ? setChildren(node, [child]) : node;
}
function pruneAggregate(node) {
  const childRequired = /* @__PURE__ */ new Set();
  for (const expr2 of node.groupBy || []) collectExprColumns(expr2, childRequired);
  for (const agg of node.aggregates || []) {
    for (const arg of agg.args || []) collectExprColumns(arg, childRequired);
  }
  const child = pruneColumns(node.children[0], childRequired);
  return child !== node.children[0] ? setChildren(node, [child]) : node;
}
function pruneSort(node, required) {
  if (!required) return pruneUnary(node, null);
  let childRequired = copyRefs(required);
  for (const key of node.orderKeys || []) childRequired = addExprRefs(childRequired, key.expr);
  return pruneUnary(node, childRequired);
}
function pruneJoin(node, required) {
  const left = node.children[0];
  const right = node.children[1];
  if (!required) {
    const newLeft2 = pruneColumns(left, null);
    const newRight2 = pruneColumns(right, null);
    if (newLeft2 !== left || newRight2 !== right) return setChildren(node, [newLeft2, newRight2]);
    return node;
  }
  const leftRefs = collectPlanRefs2(left);
  const rightRefs = collectPlanRefs2(right);
  const refs = copyRefs(required);
  if (node.condition) collectExprColumns(node.condition, refs);
  const leftRequired = filterRefsForPlan(refs, leftRefs);
  const rightRequired = filterRefsForPlan(refs, rightRefs);
  const newLeft = pruneColumns(left, leftRequired);
  const newRight = pruneColumns(right, rightRequired);
  if (newLeft !== left || newRight !== right) return setChildren(node, [newLeft, newRight]);
  return node;
}
function pruneDependentJoin(node, required) {
  const refs = copyRefs(required);
  if (node.condition) collectExprColumns(node.condition, refs);
  for (const expr2 of node.correlatedColumns || []) collectExprColumns(expr2, refs);
  const children = node.children || [];
  if (children.length !== 2) return pruneChildren(node, refs);
  const newLeft = pruneColumns(children[0], refs);
  const newRight = pruneColumns(children[1], null);
  if (newLeft !== children[0] || newRight !== children[1]) return setChildren(node, [newLeft, newRight]);
  return node;
}
function pruneUnary(node, required) {
  const child = pruneColumns(node.children?.[0], required);
  return child !== node.children?.[0] ? setChildren(node, [child]) : node;
}
function pruneChildren(node, required) {
  const children = getChildren(node);
  const newChildren = children.map((child) => pruneColumns(child, required));
  const changed = newChildren.some((child, i) => child !== children[i]);
  return changed ? setChildren(node, newChildren) : node;
}
function addExprRefs(required, expr2) {
  const refs = copyRefs(required);
  collectExprColumns(expr2, refs);
  return refs;
}
function copyRefs(required) {
  return required ? new Set(required) : /* @__PURE__ */ new Set();
}
function collectExprColumns(expr2, required) {
  if (!expr2 || typeof expr2 !== "object") return;
  if (expr2.kind === BoundExprKind.COLUMN_REF) {
    required.add(refKey(expr2.tableAlias, expr2.columnName));
    if (Number.isInteger(expr2.columnIndex) && expr2.columnIndex >= 0) {
      required.add(refKey(expr2.tableAlias, `#${expr2.columnIndex}`));
    }
    return;
  }
  for (const val of Object.values(expr2)) {
    if (Array.isArray(val)) {
      for (const item of val) collectExprColumns(item, required);
    } else if (val && typeof val === "object") {
      collectExprColumns(val, required);
    }
  }
}
function collectPlanRefs2(node) {
  const refs = { aliases: /* @__PURE__ */ new Set(), columns: /* @__PURE__ */ new Set() };
  addOutputRefs2(node, refs);
  refs.aliases.delete("");
  refs.columns.delete("");
  return refs;
}
function addOutputRefs2(node, refs) {
  if (!node) return;
  if (node.type === PlanNodeType.SCAN) {
    refs.aliases.add((node.alias || node.table || "").toUpperCase());
    for (const col2 of node.columns || []) refs.columns.add((col2.name || col2.columnName || "").toUpperCase());
    return;
  }
  if (node.type === PlanNodeType.CTE_SCAN) {
    refs.aliases.add((node.alias || node.cteName || "").toUpperCase());
    return;
  }
  if (node.type === PlanNodeType.PROJECT) {
    for (const expr2 of node.expressions || []) refs.columns.add(outputName(expr2));
    for (const child of getChildren(node)) addOutputRefs2(child, refs);
    return;
  }
  if (node.type === PlanNodeType.AGGREGATE) {
    for (const expr2 of node.groupBy || []) refs.columns.add(outputName(expr2));
    for (const agg of node.aggregates || []) refs.columns.add(outputName(agg));
    return;
  }
  if (node.type === PlanNodeType.JOIN || node.type === PlanNodeType.UNION) {
    for (const child of getChildren(node)) addOutputRefs2(child, refs);
    return;
  }
  if (node.children?.[0]) addOutputRefs2(node.children[0], refs);
}
function outputName(expr2) {
  return (expr2?.outputName || expr2?.alias || expr2?.name || expr2?.columnName || "").toUpperCase();
}
function outputNeeded(expr2, required, index) {
  const name = outputName(expr2);
  if (!name && index === void 0) return true;
  for (const ref of required) {
    const { columnName } = parseRef(ref);
    if (name && columnName === name) return true;
    if (index !== void 0 && columnName === `#${index}`) return true;
  }
  return false;
}
function filterRefsForPlan(required, planRefs) {
  const result = /* @__PURE__ */ new Set();
  for (const ref of required || []) {
    const parsed = parseRef(ref);
    if (refBelongsToPlan2(parsed, planRefs)) result.add(ref);
  }
  return result;
}
function refSetNeedsColumn(required, aliases, columnName) {
  const col2 = (columnName || "").toUpperCase();
  for (const ref of required) {
    const parsed = parseRef(ref);
    if (parsed.columnName !== col2) continue;
    if (!parsed.tableAlias || parsed.tableAlias === "." || aliases.has(parsed.tableAlias)) return true;
  }
  return false;
}
function refBelongsToPlan2(ref, planRefs) {
  if (ref.tableAlias && ref.tableAlias !== ".") return planRefs.aliases.has(ref.tableAlias);
  return planRefs.columns.has(ref.columnName);
}
function refKey(tableAlias, columnName) {
  return `${(tableAlias || "").toUpperCase()}.${(columnName || "").toUpperCase()}`;
}
function parseRef(ref) {
  const dot = ref.indexOf(".");
  if (dot < 0) return { tableAlias: "", columnName: ref.toUpperCase() };
  return {
    tableAlias: ref.slice(0, dot).toUpperCase(),
    columnName: ref.slice(dot + 1).toUpperCase()
  };
}

// src/optimizer/passes/join-reorder.js
init_buffer_shim();
init_logical_plan();

// src/optimizer/dphyp/hypergraph.js
init_buffer_shim();
init_expression_binder();
var HyperEdge = class {
  constructor(leftMask, rightMask, predicate) {
    this.leftMask = leftMask;
    this.rightMask = rightMask;
    this.predicate = predicate;
  }
};
var HyperGraph = class {
  constructor() {
    this.relations = [];
    this.relationIndex = /* @__PURE__ */ new Map();
    this.edges = [];
    this.adjacency = [];
  }
  addRelation(name, plan, cardinality) {
    const id = this.relations.length;
    if (id >= 30) return -1;
    const mask = 1 << id;
    this.relations.push({ id, name, plan, cardinality, mask });
    this.relationIndex.set(name.toUpperCase(), id);
    this.adjacency.push(0);
    return id;
  }
  addEdge(leftNames, rightNames, predicate) {
    let leftMask = 0;
    for (const name of leftNames) {
      const id = this.relationIndex.get(name.toUpperCase());
      if (id !== void 0) leftMask |= 1 << id;
    }
    let rightMask = 0;
    for (const name of rightNames) {
      const id = this.relationIndex.get(name.toUpperCase());
      if (id !== void 0) rightMask |= 1 << id;
    }
    if (leftMask === 0 || rightMask === 0) return;
    this.edges.push(new HyperEdge(leftMask, rightMask, predicate));
    for (let i = 0; i < this.relations.length; i++) {
      const bit = 1 << i;
      if (leftMask & bit) this.adjacency[i] |= rightMask;
      if (rightMask & bit) this.adjacency[i] |= leftMask;
    }
  }
  getNeighborhood(subset) {
    let neighbors = 0;
    for (let i = 0; i < this.relations.length; i++) {
      if (subset & 1 << i) {
        neighbors |= this.adjacency[i];
      }
    }
    return neighbors & ~subset;
  }
  isConnected(subset) {
    if (subset === 0) return false;
    const startBit = lowestBit(subset);
    let reached = startBit;
    let frontier = startBit;
    while (frontier !== 0) {
      let nextFrontier = 0;
      for (let i = 0; i < this.relations.length; i++) {
        if (!(frontier & 1 << i)) continue;
        const adj = this.adjacency[i] & subset & ~reached;
        nextFrontier |= adj;
        reached |= adj;
      }
      frontier = nextFrontier;
    }
    return reached === subset;
  }
  findJoinPredicates(leftMask, rightMask) {
    const preds = [];
    const seen = /* @__PURE__ */ new Set();
    for (const edge of this.edges) {
      const edgeFull = edge.leftMask | edge.rightMask;
      const combined = leftMask | rightMask;
      if ((edgeFull & combined) !== edgeFull) continue;
      const matchNormal = (edge.leftMask & leftMask) !== 0 && (edge.rightMask & rightMask) !== 0;
      const matchFlipped = (edge.leftMask & rightMask) !== 0 && (edge.rightMask & leftMask) !== 0;
      if ((matchNormal || matchFlipped) && !seen.has(edge)) {
        seen.add(edge);
        preds.push(edge.predicate);
      }
    }
    return preds;
  }
  get size() {
    return this.relations.length;
  }
  get fullMask() {
    return (1 << this.relations.length) - 1;
  }
};
function buildHyperGraph(relations, joinPredicates, cardinalityEstimator) {
  const graph = new HyperGraph();
  for (const rel of relations) {
    const card = cardinalityEstimator.estimatePlan ? cardinalityEstimator.estimatePlan(rel.plan) : cardinalityEstimator.estimateScan(rel.name);
    const id = graph.addRelation(rel.alias || rel.name, rel.plan, card);
    if (id === -1) return graph;
  }
  for (const pred of joinPredicates) {
    const refs = collectColumnTableRefs(pred);
    if (refs.size < 2) continue;
    const refsArray = [...refs];
    for (let i = 0; i < refsArray.length; i++) {
      for (let j = i + 1; j < refsArray.length; j++) {
        graph.addEdge([refsArray[i]], [refsArray[j]], pred);
      }
    }
  }
  return graph;
}
function collectColumnTableRefs(expr2) {
  const refs = /* @__PURE__ */ new Set();
  _walkExpr2(expr2, (e) => {
    if (e.kind === BoundExprKind.COLUMN_REF && e.tableAlias) {
      refs.add(e.tableAlias.toUpperCase());
    }
  });
  return refs;
}
function _walkExpr2(expr2, fn) {
  if (!expr2 || typeof expr2 !== "object") return;
  fn(expr2);
  if (expr2.left) _walkExpr2(expr2.left, fn);
  if (expr2.right) _walkExpr2(expr2.right, fn);
  if (expr2.operand) _walkExpr2(expr2.operand, fn);
  if (expr2.args) for (const a of expr2.args) _walkExpr2(a, fn);
}
function lowestBit(mask) {
  return mask & -mask;
}
function popcount(mask) {
  let count2 = 0;
  while (mask) {
    count2 += mask & 1;
    mask >>>= 1;
  }
  return count2;
}

// src/optimizer/dphyp/dphyp.js
init_buffer_shim();
init_expression_binder();
var DPhypEnumerator = class {
  constructor(hyperGraph, costModel, cardinalityEstimator) {
    this.graph = hyperGraph;
    this.costModel = costModel;
    this.cardEstimator = cardinalityEstimator;
    this.dp = /* @__PURE__ */ new Map();
  }
  solve() {
    for (const rel of this.graph.relations) {
      const cost = this.costModel.scanCost(rel.cardinality);
      this.dp.set(rel.mask, {
        plan: rel.plan,
        cardinality: rel.cardinality,
        totalCost: cost,
        mask: rel.mask
      });
    }
    const n = this.graph.size;
    for (let size = 2; size <= n; size++) {
      this.enumerateSize(size);
    }
    return this.dp.get(this.graph.fullMask) || null;
  }
  enumerateSize(size) {
    const fullMask = this.graph.fullMask;
    for (let mask = 1; mask <= fullMask; mask++) {
      if (popcount(mask) !== size) continue;
      if (this.dp.has(mask)) continue;
      if (!this.graph.isConnected(mask)) continue;
      this.enumerateCsgCmpPairs(mask);
    }
  }
  enumerateCsgCmpPairs(combined) {
    for (let s1 = combined - 1 & combined; s1 > 0; s1 = s1 - 1 & combined) {
      const s2 = combined & ~s1;
      if (s2 === 0) continue;
      if (s1 > s2) continue;
      if (!this.dp.has(s1) || !this.dp.has(s2)) continue;
      if (!this.graph.isConnected(s1) || !this.graph.isConnected(s2)) continue;
      const preds = this.graph.findJoinPredicates(s1, s2);
      if (preds.length === 0) continue;
      this.emitPair(s1, s2, combined, preds);
    }
  }
  emitPair(s1Mask, s2Mask, combinedMask, predicates) {
    const s1 = this.dp.get(s1Mask);
    const s2 = this.dp.get(s2Mask);
    const joinCondition = combinePredicates(predicates);
    this.tryJoinOrder(s1, s2, combinedMask, joinCondition);
    this.tryJoinOrder(s2, s1, combinedMask, joinCondition);
  }
  tryJoinOrder(build, probe, combinedMask, joinCondition) {
    const joinCard = this.cardEstimator.estimateJoin(
      build.cardinality,
      probe.cardinality,
      joinCondition
    );
    const buildCard = build.cardinality;
    const probeCard = probe.cardinality;
    const joinCost = build.totalCost + probe.totalCost + this.costModel.hashJoinCost(buildCard, probeCard, joinCard);
    const existing = this.dp.get(combinedMask);
    if (!existing || joinCost < existing.totalCost) {
      this.dp.set(combinedMask, {
        plan: {
          type: "HashJoin",
          buildSide: build.plan,
          probeSide: probe.plan,
          condition: joinCondition,
          buildCard: build.cardinality,
          probeCard: probe.cardinality
        },
        cardinality: joinCard,
        totalCost: joinCost,
        mask: combinedMask
      });
    }
  }
};
function combinePredicates(preds) {
  if (preds.length === 0) return null;
  if (preds.length === 1) return preds[0];
  const unique = [];
  const seen = /* @__PURE__ */ new Set();
  for (const p of preds) {
    const key = predicateKey(p);
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(p);
    }
  }
  if (unique.length === 1) return unique[0];
  return unique.reduce((acc, p) => ({
    kind: BoundExprKind.BINARY,
    op: "AND",
    left: acc,
    right: p,
    resultType: "BOOLEAN"
  }));
}
function predicateKey(pred) {
  if (!pred) return "";
  if (pred.kind === BoundExprKind.COLUMN_REF) {
    return `${pred.tableAlias}.${pred.columnName}`;
  }
  if (pred.kind === BoundExprKind.BINARY) {
    return `${predicateKey(pred.left)}${pred.op}${predicateKey(pred.right)}`;
  }
  if (pred.kind === BoundExprKind.LITERAL) {
    return String(pred.value);
  }
  return JSON.stringify(pred).slice(0, 50);
}
function runDPhyp(hyperGraph, costModel, cardinalityEstimator) {
  const enumerator = new DPhypEnumerator(hyperGraph, costModel, cardinalityEstimator);
  return enumerator.solve();
}

// src/optimizer/dphyp/cost-model.js
init_buffer_shim();
var DefaultCostModel = class {
  constructor(options = {}) {
    this.C_HASH_BUILD = options.hashBuildCost ?? 1.5;
    this.C_HASH_PROBE = options.hashProbeCost ?? 1;
    this.C_COMPARE = options.compareCost ?? 0.3;
    this.C_SCAN = options.scanCost ?? 0.1;
    this.C_FILTER = options.filterCost ?? 0.2;
    this.C_OUTPUT = options.outputCost ?? 0.3;
    this.C_MEMORY = options.memoryCost ?? 0.05;
    this.C_IO = options.ioCost ?? 5;
    this.SPILL_THRESHOLD = options.spillThreshold ?? 2e5;
    this.C_CROSS = options.crossJoinPenalty ?? 1e3;
  }
  hashJoinCost(buildCard, probeCard, outputCard = null) {
    const cpu = buildCard * this.C_HASH_BUILD + probeCard * this.C_HASH_PROBE;
    const mem = buildCard * this.C_MEMORY;
    const out = (outputCard || Math.max(buildCard, probeCard)) * this.C_OUTPUT;
    const spill = buildCard > this.SPILL_THRESHOLD ? (buildCard + probeCard) * this.C_IO : 0;
    return cpu + mem + out + spill;
  }
  mergeJoinCost(leftCard, rightCard, outputCard = null) {
    const cpu = (leftCard + rightCard) * this.C_COMPARE;
    const out = (outputCard || Math.max(leftCard, rightCard)) * this.C_OUTPUT;
    return cpu + out;
  }
  sortMergeJoinCost(leftCard, rightCard, outputCard = null) {
    return this.sortCost(leftCard) + this.sortCost(rightCard) + this.mergeJoinCost(leftCard, rightCard, outputCard);
  }
  nestedLoopJoinCost(outerCard, innerCard) {
    return outerCard * innerCard * this.C_COMPARE;
  }
  crossJoinCost(leftCard, rightCard) {
    return leftCard * rightCard * this.C_CROSS;
  }
  sortCost(card) {
    if (card <= 1) return 0;
    return card * Math.log2(card) * this.C_COMPARE;
  }
  topNSortCost(card, limit) {
    if (card <= 1 || limit <= 0) return 0;
    const k = Math.min(limit, card);
    return card * Math.log2(Math.max(2, k)) * this.C_COMPARE;
  }
  scanCost(card) {
    return card * this.C_SCAN;
  }
  filterCost(card) {
    return card * this.C_FILTER;
  }
  aggregateCost(card) {
    return this.hashAggregateCost(card);
  }
  hashAggregateCost(card, numGroups = null) {
    const groups = numGroups || Math.max(1, Math.sqrt(card));
    return card * this.C_HASH_BUILD + groups * this.C_MEMORY;
  }
  streamAggregateCost(card) {
    return card * this.C_SCAN;
  }
  totalJoinCost(buildPlan, probePlan, buildCard, probeCard, outputCard) {
    return buildPlan.totalCost + probePlan.totalCost + this.hashJoinCost(buildCard, probeCard, outputCard);
  }
  cheaperJoinCost(leftCard, rightCard, leftSorted, rightSorted, outputCard, downstreamSortSaving = 0) {
    const buildCard = Math.min(leftCard, rightCard);
    const probeCard = Math.max(leftCard, rightCard);
    const hashCost = this.hashJoinCost(buildCard, probeCard, outputCard);
    const leftSortCost = leftSorted ? 0 : this.sortCost(leftCard);
    const rightSortCost = rightSorted ? 0 : this.sortCost(rightCard);
    const mergeCost = leftSortCost + rightSortCost + this.mergeJoinCost(leftCard, rightCard, outputCard) - downstreamSortSaving;
    return { hashCost, mergeCost, preferMerge: mergeCost < hashCost };
  }
};

// src/optimizer/dphyp/cardinality.js
init_buffer_shim();
init_expression_binder();
init_logical_plan();
var MIN_SELECTIVITY = 1e-4;
var DEFAULT_CORRELATION = 0.5;
var DEFAULT_NDV = 100;
var DEFAULT_SCAN_ROWS = 1e3;
var DefaultCardinalityEstimator = class {
  constructor(statisticsProvider) {
    this.stats = statisticsProvider;
  }
  estimateScan(tableName) {
    const tableStats = this.stats.get(tableName.toUpperCase());
    return tableStats ? tableStats.rowCount : DEFAULT_SCAN_ROWS;
  }
  estimatePlan(node) {
    if (!node) return DEFAULT_SCAN_ROWS;
    switch (node.type) {
      case PlanNodeType.SCAN:
        return this.estimateScan(node.table);
      case PlanNodeType.FILTER:
        return this.estimateFilter(this.estimatePlan(node.children[0]), node.condition);
      case PlanNodeType.PROJECT:
      case PlanNodeType.SORT:
      case PlanNodeType.DISTINCT:
      case PlanNodeType.MATERIALIZE:
        return this.estimatePlan(node.children?.[0]);
      case PlanNodeType.LIMIT: {
        const childCard = this.estimatePlan(node.children?.[0]);
        return Math.min(node.count || childCard, childCard);
      }
      case PlanNodeType.JOIN: {
        const leftCard = this.estimatePlan(node.children[0]);
        const rightCard = this.estimatePlan(node.children[1]);
        if (node.joinType === JoinType.SEMI) return this.estimateSemiJoin(leftCard, rightCard, node.condition);
        if (node.joinType === JoinType.ANTI) return this.estimateAntiJoin(leftCard, rightCard, node.condition);
        if (node.joinType === JoinType.MARK) return leftCard;
        if (node.joinType === JoinType.LEFT) return this.estimateLeftJoin(leftCard, rightCard, node.condition);
        if (node.joinType === JoinType.CROSS) return leftCard * rightCard;
        return this.estimateJoin(leftCard, rightCard, node.condition);
      }
      case PlanNodeType.AGGREGATE:
        return this.estimateAggregate(this.estimatePlan(node.children[0]), node.groupBy?.length || 0, node.groupBy);
      case PlanNodeType.EMPTY:
        return 0;
      default:
        return node.children?.length ? this.estimatePlan(node.children[0]) : DEFAULT_SCAN_ROWS;
    }
  }
  estimateFilter(inputCard, predicate) {
    const sel = this.estimateSelectivity(predicate);
    return Math.max(1, Math.round(inputCard * sel));
  }
  estimateJoin(leftCard, rightCard, condition) {
    if (!condition) return leftCard * rightCard;
    const equiPreds = this.extractEquiPredicates(condition);
    if (equiPreds.length === 0) {
      const sel = this.estimateSelectivity(condition);
      return Math.max(1, Math.round(leftCard * rightCard * sel));
    }
    const leftNdvs = [];
    const rightNdvs = [];
    for (const pred of equiPreds) {
      leftNdvs.push(this.getColumnNdv(pred.left));
      rightNdvs.push(this.getColumnNdv(pred.right));
    }
    const combinedLeftNdv = Math.min(
      leftNdvs.reduce((acc, n) => acc * n, 1),
      leftCard
    );
    const combinedRightNdv = Math.min(
      rightNdvs.reduce((acc, n) => acc * n, 1),
      rightCard
    );
    const divisor = Math.max(combinedLeftNdv, combinedRightNdv, 1);
    return Math.max(1, Math.round(leftCard * rightCard / divisor));
  }
  estimateLeftJoin(leftCard, rightCard, condition) {
    const innerCard = this.estimateJoin(leftCard, rightCard, condition);
    return Math.max(leftCard, innerCard);
  }
  estimateSemiJoin(leftCard, rightCard, condition) {
    if (!condition) return Math.round(leftCard * 0.5);
    const equiPreds = this.extractEquiPredicates(condition);
    if (equiPreds.length > 0) {
      let selectivity = 1;
      for (const pred of equiPreds) {
        const leftNdv = this.getColumnNdv(pred.left);
        const rightNdv = this.getColumnNdv(pred.right);
        selectivity = Math.min(selectivity, Math.min(1, rightNdv / Math.max(leftNdv, 1)));
      }
      return Math.max(1, Math.round(leftCard * selectivity));
    }
    return Math.max(1, Math.round(leftCard * 0.5));
  }
  estimateAntiJoin(leftCard, rightCard, condition) {
    const semiCard = this.estimateSemiJoin(leftCard, rightCard, condition);
    return Math.max(1, leftCard - semiCard);
  }
  estimateAggregate(inputCard, groupByCount, groupByExprs = []) {
    if (groupByCount === 0) return 1;
    let ndvProduct = 1;
    const ndvs = (groupByExprs || []).map((expr2) => this.getColumnNdv(expr2)).sort((a, b) => b - a);
    for (let i = 0; i < ndvs.length; i++) {
      if (i === 0) {
        ndvProduct = ndvs[i];
      } else {
        ndvProduct *= Math.max(1, Math.sqrt(ndvs[i]));
      }
    }
    if (ndvProduct > 1) {
      return Math.max(1, Math.min(inputCard, Math.round(ndvProduct)));
    }
    return Math.min(inputCard, Math.pow(10, groupByCount));
  }
  estimateSelectivity(predicate) {
    if (!predicate) return 1;
    switch (predicate.kind) {
      case BoundExprKind.BINARY: {
        if (predicate.op === "AND") {
          const sl = this.estimateSelectivity(predicate.left);
          const sr = this.estimateSelectivity(predicate.right);
          const independent = sl * sr;
          const correlated = Math.min(sl, sr);
          const correlation = this.lookupCorrelation(predicate.left, predicate.right);
          const blended = independent * (1 - correlation) + correlated * correlation;
          return Math.max(MIN_SELECTIVITY, Math.min(correlated, blended));
        }
        if (predicate.op === "OR") {
          const sl = this.estimateSelectivity(predicate.left);
          const sr = this.estimateSelectivity(predicate.right);
          return Math.min(1, sl + sr - sl * sr);
        }
        if (predicate.op === "=") return this.estimateEqualitySelectivity(predicate);
        if (["<", ">", "<=", ">="].includes(predicate.op)) return this.estimateRangeSelectivity(predicate);
        if (predicate.op === "<>") {
          const eqSel = this.estimateEqualitySelectivity({ ...predicate, op: "=" });
          return Math.max(MIN_SELECTIVITY, 1 - eqSel);
        }
        return 0.5;
      }
      case BoundExprKind.BETWEEN:
        return this.estimateBetweenSelectivity(predicate);
      case BoundExprKind.LIKE:
        return this.estimateLikeSelectivity(predicate);
      case BoundExprKind.IN_LIST:
        return this.estimateInListSelectivity(predicate);
      case BoundExprKind.IS_NULL:
        return this.estimateIsNullSelectivity(predicate);
      case BoundExprKind.UNARY:
        if (predicate.op === "NOT") return Math.max(MIN_SELECTIVITY, 1 - this.estimateSelectivity(predicate.operand));
        return 0.5;
      case BoundExprKind.EXISTS:
        return 0.5;
      default:
        return 0.5;
    }
  }
  estimateEqualitySelectivity(predicate) {
    let column = null, literal = null;
    if (predicate.left?.kind === BoundExprKind.COLUMN_REF && predicate.right?.kind === BoundExprKind.LITERAL) {
      column = predicate.left;
      literal = predicate.right;
    } else if (predicate.right?.kind === BoundExprKind.COLUMN_REF && predicate.left?.kind === BoundExprKind.LITERAL) {
      column = predicate.right;
      literal = predicate.left;
    } else if (predicate.left?.kind === BoundExprKind.COLUMN_REF && predicate.right?.kind === BoundExprKind.COLUMN_REF) {
      const leftNdv = this.getColumnNdv(predicate.left);
      const rightNdv = this.getColumnNdv(predicate.right);
      return 1 / Math.max(leftNdv, rightNdv, 1);
    } else {
      return 0.1;
    }
    const stats = this.getColumnStats(column);
    if (!stats) {
      return 0.1;
    }
    if (stats.mcv) {
      const litStr = String(literal.value);
      const mcvIdx = stats.mcv.values.indexOf(litStr);
      if (mcvIdx >= 0) {
        return stats.mcv.frequencies[mcvIdx] * (1 - stats.nullFraction);
      }
    }
    const ndv = stats.ndv || DEFAULT_NDV;
    const nullFrac = stats.nullFraction || 0;
    return Math.max(MIN_SELECTIVITY, (1 - nullFrac) / ndv);
  }
  estimateRangeSelectivity(predicate) {
    const column = predicate.left?.kind === BoundExprKind.COLUMN_REF ? predicate.left : predicate.right;
    const literal = predicate.left?.kind === BoundExprKind.LITERAL ? predicate.left : predicate.right;
    if (column?.kind !== BoundExprKind.COLUMN_REF || literal?.kind !== BoundExprKind.LITERAL) {
      return 0.33;
    }
    const stats = this.getColumnStats(column);
    if (!stats) return 0.33;
    if (stats.histogram) {
      const isLessThan = predicate.left === column ? ["<", "<="].includes(predicate.op) : [">", ">="].includes(predicate.op);
      const frac = stats.histogram.estimateLessThan(literal.value);
      const sel = isLessThan ? frac : 1 - frac;
      return Math.max(MIN_SELECTIVITY, sel * (1 - (stats.nullFraction || 0)));
    }
    const min2 = toNumber(stats.min);
    const max2 = toNumber(stats.max);
    const value = toNumber(literal.value);
    if (min2 === null || max2 === null || value === null || max2 <= min2) return 0.33;
    const ratio = (value - min2) / (max2 - min2);
    const clamped = Math.max(0, Math.min(1, ratio));
    const lessThan = predicate.left === column ? ["<", "<="].includes(predicate.op) : [">", ">="].includes(predicate.op);
    return Math.max(MIN_SELECTIVITY, (lessThan ? clamped : 1 - clamped) * (1 - (stats.nullFraction || 0)));
  }
  estimateBetweenSelectivity(predicate) {
    const stats = this.getColumnStats(predicate.expr);
    if (!stats) return 0.25;
    if (stats.histogram && predicate.low?.kind === BoundExprKind.LITERAL && predicate.high?.kind === BoundExprKind.LITERAL) {
      const sel2 = stats.histogram.estimateRange(predicate.low.value, predicate.high.value);
      const result = predicate.negated ? 1 - sel2 : sel2;
      return Math.max(MIN_SELECTIVITY, result * (1 - (stats.nullFraction || 0)));
    }
    const min2 = toNumber(stats.min);
    const max2 = toNumber(stats.max);
    const low = toNumber(predicate.low?.value);
    const high = toNumber(predicate.high?.value);
    if (min2 === null || max2 === null || low === null || high === null || max2 <= min2) return 0.25;
    const covered = Math.max(0, Math.min(max2, high) - Math.max(min2, low));
    const sel = covered / (max2 - min2);
    return Math.max(MIN_SELECTIVITY, Math.min(1, predicate.negated ? 1 - sel : sel));
  }
  estimateInListSelectivity(predicate) {
    if (Array.isArray(predicate.list)) {
      const ndv = this.getColumnNdv(predicate.expr);
      const stats = this.getColumnStats(predicate.expr);
      const nullFrac = stats?.nullFraction ?? 0;
      if (stats?.mcv) {
        let mcvHits = 0;
        let mcvFreq = 0;
        for (const item of predicate.list) {
          if (item.kind === BoundExprKind.LITERAL) {
            const idx = stats.mcv.values.indexOf(String(item.value));
            if (idx >= 0) {
              mcvHits++;
              mcvFreq += stats.mcv.frequencies[idx];
            }
          }
        }
        const nonMcvItems = predicate.list.length - mcvHits;
        const nonMcvNdv = Math.max(1, ndv - stats.mcv.values.length);
        const nonMcvFreq = nonMcvItems / nonMcvNdv;
        const totalSel = Math.min(1, (mcvFreq + nonMcvFreq) * (1 - nullFrac));
        return predicate.negated ? Math.max(MIN_SELECTIVITY, 1 - totalSel) : Math.max(MIN_SELECTIVITY, totalSel);
      }
      const sel = Math.min(1, predicate.list.length / Math.max(ndv, 1)) * (1 - nullFrac);
      return predicate.negated ? Math.max(MIN_SELECTIVITY, 1 - sel) : Math.max(MIN_SELECTIVITY, sel);
    }
    return predicate.negated ? 0.7 : 0.3;
  }
  estimateIsNullSelectivity(predicate) {
    const stats = this.getColumnStats(predicate.expr);
    const nullFrac = stats?.nullFraction || 0.05;
    return predicate.negated ? Math.max(MIN_SELECTIVITY, 1 - nullFrac) : Math.max(MIN_SELECTIVITY, nullFrac);
  }
  estimateLikeSelectivity(predicate) {
    const pattern = predicate.pattern?.value;
    if (typeof pattern !== "string") return this.likeFallback("unknown");
    const ndv = this.getColumnNdv(predicate.expr);
    const stats = this.getColumnStats(predicate.expr);
    const avgLen = stats?.avgLength;
    if (!pattern.includes("%") && !pattern.includes("_")) {
      return ndv > 0 ? 1 / ndv : this.likeFallback("exact");
    }
    if (this.isPrefixPattern(pattern)) {
      const prefixLen = pattern.length - 1;
      if (avgLen && avgLen > 0) {
        const exponent = Math.min(prefixLen / avgLen, 1);
        return Math.max(MIN_SELECTIVITY, Math.pow(ndv, -exponent));
      }
      return Math.max(MIN_SELECTIVITY, 1 / Math.pow(ndv, Math.min(1, prefixLen / 4)));
    }
    if (this.isContainsPattern(pattern)) {
      const inner = pattern.slice(1, -1);
      if (avgLen && avgLen > 0) {
        const coverageRatio = Math.min(1, inner.length / avgLen);
        const baseSelectivity = 1 / Math.max(Math.sqrt(ndv), 1);
        return Math.max(MIN_SELECTIVITY, baseSelectivity * (1 - coverageRatio * 0.5));
      }
      return Math.max(MIN_SELECTIVITY, 1 / Math.max(Math.sqrt(ndv), 1));
    }
    if (this.isSuffixPattern(pattern)) {
      const suffixLen = pattern.length - 1;
      if (avgLen && avgLen > 0) {
        return Math.max(MIN_SELECTIVITY, 1 / Math.pow(ndv, Math.min(1, suffixLen / avgLen)));
      }
      return Math.max(MIN_SELECTIVITY, 1 / Math.max(Math.sqrt(ndv), 1));
    }
    return this.likeFallback("complex");
  }
  isPrefixPattern(pattern) {
    return pattern.endsWith("%") && !pattern.slice(0, -1).includes("%") && !pattern.includes("_");
  }
  isContainsPattern(pattern) {
    return pattern.startsWith("%") && pattern.endsWith("%") && !pattern.slice(1, -1).includes("%");
  }
  isSuffixPattern(pattern) {
    return pattern.startsWith("%") && !pattern.slice(1).includes("%") && !pattern.includes("_");
  }
  likeFallback(type) {
    const fallbacks = { exact: 0.1, unknown: 0.1, complex: 0.15 };
    return fallbacks[type] ?? 0.1;
  }
  lookupCorrelation(leftPred, rightPred) {
    const leftCol = this.extractSingleColumn(leftPred);
    const rightCol = this.extractSingleColumn(rightPred);
    if (!leftCol || !rightCol) return DEFAULT_CORRELATION;
    if (leftCol.tableAlias?.toUpperCase() !== rightCol.tableAlias?.toUpperCase()) return DEFAULT_CORRELATION;
    const tableStats = this.stats.get(leftCol.tableAlias.toUpperCase());
    if (!tableStats?.getCorrelation) return DEFAULT_CORRELATION;
    const corr = tableStats.getCorrelation(leftCol.columnName, rightCol.columnName);
    return corr !== null ? Math.abs(corr) : DEFAULT_CORRELATION;
  }
  extractSingleColumn(pred) {
    if (pred?.kind === BoundExprKind.COLUMN_REF) return pred;
    if (pred?.kind === BoundExprKind.BINARY && ["=", "<", ">", "<=", ">=", "<>"].includes(pred.op)) {
      if (pred.left?.kind === BoundExprKind.COLUMN_REF) return pred.left;
      if (pred.right?.kind === BoundExprKind.COLUMN_REF) return pred.right;
    }
    if (pred?.kind === BoundExprKind.BETWEEN && pred.expr?.kind === BoundExprKind.COLUMN_REF) return pred.expr;
    if (pred?.kind === BoundExprKind.LIKE && pred.expr?.kind === BoundExprKind.COLUMN_REF) return pred.expr;
    if (pred?.kind === BoundExprKind.IN_LIST && pred.expr?.kind === BoundExprKind.COLUMN_REF) return pred.expr;
    return null;
  }
  getColumnNdv(expr2) {
    if (expr2?.kind !== BoundExprKind.COLUMN_REF) return DEFAULT_NDV;
    const colStats = this.getColumnStats(expr2);
    return colStats?.ndv || DEFAULT_NDV;
  }
  getColumnStats(expr2) {
    if (!expr2) return null;
    if (expr2.kind !== BoundExprKind.COLUMN_REF) return null;
    const columnName = expr2.columnName?.toUpperCase();
    const tableStats = this.stats.get(expr2.tableAlias?.toUpperCase());
    if (tableStats?.columnStats?.has(columnName)) {
      return tableStats.columnStats.get(columnName);
    }
    for (const stats of this.stats.values()) {
      if (stats.columnStats?.has(columnName)) {
        return stats.columnStats.get(columnName);
      }
    }
    return null;
  }
  extractEquiPredicates(condition) {
    const result = [];
    this._collectEqui(condition, result);
    return result;
  }
  _collectEqui(expr2, result) {
    if (!expr2) return;
    if (expr2.kind === BoundExprKind.BINARY && expr2.op === "AND") {
      this._collectEqui(expr2.left, result);
      this._collectEqui(expr2.right, result);
      return;
    }
    if (expr2.kind === BoundExprKind.BINARY && expr2.op === "=" && expr2.left?.kind === BoundExprKind.COLUMN_REF && expr2.right?.kind === BoundExprKind.COLUMN_REF) {
      result.push({ left: expr2.left, right: expr2.right });
    }
  }
};
function toNumber(value) {
  if (value === null || value === void 0) return null;
  if (typeof value === "bigint") return Number(value);
  if (typeof value === "number") return value;
  return null;
}

// src/optimizer/passes/join-reorder.js
init_expression_binder();
var JoinReorder = class extends OptimizationPass {
  constructor(statisticsMap = /* @__PURE__ */ new Map(), costModel = null, cardEstimator = null) {
    super();
    this.statisticsMap = statisticsMap;
    this.costModel = costModel || new DefaultCostModel();
    this.cardEstimator = cardEstimator || new DefaultCardinalityEstimator(this.statisticsMap);
  }
  get name() {
    return "JoinReorder";
  }
  apply(plan) {
    const rewriter = new JoinReorderRewriter(this.costModel, this.cardEstimator);
    return rewriter.rewrite(plan);
  }
};
var JoinReorderRewriter = class extends PlanRewriter {
  constructor(costModel, cardEstimator) {
    super();
    this.costModel = costModel;
    this.cardEstimator = cardEstimator;
  }
  rewriteJoin(node) {
    const rewritten = this.rewriteChildren(node);
    if (this.isInnerJoinTree(rewritten)) {
      return this.reorderJoinTree(rewritten);
    }
    if (this.isNonInnerJoin(rewritten)) {
      return this.reorderNonInnerJoin(rewritten);
    }
    return rewritten;
  }
  rewriteDefault(node) {
    const rewritten = this.rewriteChildren(node);
    if (this.isInnerJoinTree(rewritten)) {
      return this.reorderJoinTree(rewritten);
    }
    return rewritten;
  }
  isInnerJoinTree(node) {
    if (node.type !== PlanNodeType.JOIN) return false;
    if (node.joinType !== JoinType.INNER && node.joinType !== JoinType.CROSS) return false;
    return true;
  }
  isNonInnerJoin(node) {
    if (node.type !== PlanNodeType.JOIN) return false;
    return node.joinType === JoinType.SEMI || node.joinType === JoinType.ANTI || node.joinType === JoinType.LEFT || node.joinType === JoinType.MARK;
  }
  reorderNonInnerJoin(node) {
    let left = node.children[0];
    let right = node.children[1];
    if (this.isInnerJoinTree(left)) left = this.reorderJoinTree(left);
    if (this.isInnerJoinTree(right)) right = this.reorderJoinTree(right);
    return { ...node, children: [left, right] };
  }
  reorderJoinTree(root) {
    const relations = [];
    const joinPredicates = [];
    const nonJoinFilters = [];
    this.flattenJoinTree(root, relations, joinPredicates, nonJoinFilters);
    if (relations.length < 2) return root;
    const graph = buildHyperGraph(relations, joinPredicates, this.cardEstimator);
    if (graph.size < 2) return root;
    const result = runDPhyp(graph, this.costModel, this.cardEstimator);
    if (!result) return root;
    let plan = this.reconstructPlan(result.plan);
    if (nonJoinFilters.length > 0) {
      plan = LogicalFilter(combineConjuncts(nonJoinFilters), plan);
    }
    return plan;
  }
  flattenJoinTree(node, relations, joinPredicates, nonJoinFilters) {
    if (node.type === PlanNodeType.JOIN && (node.joinType === JoinType.INNER || node.joinType === JoinType.CROSS)) {
      this.flattenJoinTree(node.children[0], relations, joinPredicates, nonJoinFilters);
      this.flattenJoinTree(node.children[1], relations, joinPredicates, nonJoinFilters);
      if (node.condition) {
        const preds = splitConjuncts(node.condition);
        for (const pred of preds) {
          const refs = this.collectTableRefs(pred);
          if (refs.size >= 2) {
            joinPredicates.push(pred);
          } else if (refs.size === 1) {
            nonJoinFilters.push(pred);
          } else {
            nonJoinFilters.push(pred);
          }
        }
      }
      return;
    }
    if (node.type === PlanNodeType.FILTER) {
      const preds = splitConjuncts(node.condition);
      const child = node.children[0];
      if (child.type === PlanNodeType.JOIN && (child.joinType === JoinType.INNER || child.joinType === JoinType.CROSS)) {
        for (const pred of preds) {
          const refs = this.collectTableRefs(pred);
          if (refs.size >= 2) {
            joinPredicates.push(pred);
          } else {
            nonJoinFilters.push(pred);
          }
        }
        this.flattenJoinTree(child, relations, joinPredicates, nonJoinFilters);
        return;
      }
    }
    if (node.type === PlanNodeType.FILTER && node.children[0].type === PlanNodeType.SCAN) {
      const scan = node.children[0];
      relations.push({
        name: scan.table,
        alias: scan.alias || scan.table,
        plan: node
      });
      return;
    }
    if (node.type === PlanNodeType.SCAN) {
      relations.push({
        name: node.table,
        alias: node.alias || node.table,
        plan: node
      });
      return;
    }
    const alias = this.inferAlias(node);
    relations.push({
      name: alias,
      alias,
      plan: node
    });
  }
  collectTableRefs(expr2) {
    const refs = /* @__PURE__ */ new Set();
    this._walkExpr(expr2, (e) => {
      if (e.kind === BoundExprKind.COLUMN_REF && e.tableAlias) {
        refs.add(e.tableAlias.toUpperCase());
      }
    });
    return refs;
  }
  _walkExpr(expr2, fn) {
    if (!expr2 || typeof expr2 !== "object") return;
    fn(expr2);
    if (expr2.left) this._walkExpr(expr2.left, fn);
    if (expr2.right) this._walkExpr(expr2.right, fn);
    if (expr2.operand) this._walkExpr(expr2.operand, fn);
    if (expr2.args) for (const a of expr2.args) this._walkExpr(a, fn);
  }
  reconstructPlan(dpPlan) {
    if (!dpPlan) return dpPlan;
    if (dpPlan.type === "HashJoin") {
      const left = this.reconstructPlan(dpPlan.buildSide);
      const right = this.reconstructPlan(dpPlan.probeSide);
      return LogicalJoin(JoinType.INNER, dpPlan.condition, left, right);
    }
    return dpPlan;
  }
  inferAlias(node) {
    if (node.alias) return node.alias;
    if (node.table) return node.table;
    const scan = this.findFirstScan(node);
    return scan?.alias || scan?.table || `_rel_${Math.random().toString(36).slice(2, 6)}`;
  }
  findFirstScan(node) {
    if (!node) return null;
    if (node.type === PlanNodeType.SCAN) return node;
    for (const child of getChildren(node)) {
      const found = this.findFirstScan(child);
      if (found) return found;
    }
    return null;
  }
};

// src/optimizer/passes/physical-design.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var DEFAULT_CARDINALITY = 1e3;
var PhysicalDesign = class extends OptimizationPass {
  constructor(statisticsMap = /* @__PURE__ */ new Map(), costModel = null, cardEstimator = null) {
    super();
    this.statisticsMap = statisticsMap;
    this.costModel = costModel || new DefaultCostModel();
    this.cardEstimator = cardEstimator || new DefaultCardinalityEstimator(this.statisticsMap);
  }
  get name() {
    return "PhysicalDesign";
  }
  apply(plan) {
    const rewriter = new PhysicalDesignRewriter(this.costModel, this.cardEstimator);
    return rewriter.rewrite(plan);
  }
};
var PhysicalDesignRewriter = class extends PlanRewriter {
  constructor(costModel, cardEstimator) {
    super();
    this.costModel = costModel;
    this.cardEstimator = cardEstimator;
    this.parentMap = null;
  }
  rewrite(plan) {
    this.parentMap = buildParentMap(plan);
    return super.rewrite(plan);
  }
  rewriteDefault(node) {
    const newNode = this.rewriteChildren(node);
    newNode._cardinality = this.estimateNodeCardinality(newNode);
    newNode._sortedBy = this.inferSortOrder(newNode);
    return newNode;
  }
  rewriteJoin(node) {
    const originalNode = node;
    const newNode = this.rewriteChildren(node);
    newNode._cardinality = this.estimateNodeCardinality(newNode);
    const left = newNode.children[0];
    const right = newNode.children[1];
    const leftCard = left._cardinality || DEFAULT_CARDINALITY;
    const rightCard = right._cardinality || DEFAULT_CARDINALITY;
    if (newNode.joinType !== JoinType.CROSS && newNode.condition) {
      const joinKeys = this.extractEquiJoinKeys(newNode.condition);
      if (joinKeys.leftKeys.length > 0 && joinKeys.rightKeys.length > 0) {
        const leftSorted = this.isSortedBy(left._sortedBy, joinKeys.leftKeys);
        const rightSorted = this.isSortedBy(right._sortedBy, joinKeys.rightKeys);
        const downstreamSortSaving = this.estimateDownstreamSortSaving(
          originalNode,
          joinKeys.leftKeys,
          joinKeys.rightKeys,
          newNode._cardinality
        );
        const comparison = this.costModel.cheaperJoinCost(
          leftCard,
          rightCard,
          leftSorted,
          rightSorted,
          newNode._cardinality,
          downstreamSortSaving
        );
        if (comparison.preferMerge) {
          newNode.physicalStrategy = PhysicalStrategy.MERGE;
          newNode._sortedBy = [...joinKeys.leftKeys, ...joinKeys.rightKeys];
          newNode._requiresSort = { left: !leftSorted, right: !rightSorted };
          this.assignBuildSide(newNode, leftCard, rightCard);
          return newNode;
        }
      }
    }
    this.assignBuildSide(newNode, leftCard, rightCard);
    const outerCard = newNode._buildSide === "left" ? leftCard : rightCard;
    const innerCard = newNode._buildSide === "left" ? rightCard : leftCard;
    const nlCost = this.costModel.nestedLoopJoinCost(outerCard, innerCard);
    const hashCost = this.costModel.hashJoinCost(outerCard, innerCard, newNode._cardinality);
    if (nlCost < hashCost) {
      newNode.physicalStrategy = PhysicalStrategy.NESTED_LOOP;
      newNode._sortedBy = [];
      return newNode;
    }
    if (this.isSpecialJoinType(newNode.joinType) && this.isPureEquiJoin(newNode.condition)) {
      newNode._dedupeBuild = true;
    }
    this.assignBuildSide(newNode, leftCard, rightCard);
    newNode.physicalStrategy = PhysicalStrategy.HASH;
    newNode._sortedBy = [];
    return newNode;
  }
  assignBuildSide(node, leftCard, rightCard) {
    switch (node.joinType) {
      case JoinType.LEFT:
      case JoinType.SEMI:
      case JoinType.ANTI:
      case JoinType.MARK:
        node._buildSide = "right";
        break;
      case JoinType.RIGHT:
        node._buildSide = "left";
        break;
      case JoinType.INNER:
        node._buildSide = rightCard < leftCard ? "right" : "left";
        break;
    }
  }
  isSpecialJoinType(joinType) {
    return joinType === JoinType.SEMI || joinType === JoinType.ANTI || joinType === JoinType.MARK;
  }
  rewriteAggregate(node) {
    const newNode = this.rewriteChildren(node);
    newNode._cardinality = this.estimateNodeCardinality(newNode);
    const child = newNode.children[0];
    if (newNode.groupBy && newNode.groupBy.length > 0) {
      const groupKeys = newNode.groupBy.map((g) => this.getColumnKey(g));
      const isSorted = this.isSortedByPrefix(child._sortedBy, groupKeys);
      if (isSorted) {
        const hashCost = this.costModel.hashAggregateCost(child._cardinality);
        const streamCost = this.costModel.streamAggregateCost(child._cardinality);
        if (streamCost <= hashCost) {
          newNode.physicalStrategy = PhysicalStrategy.STREAM;
          newNode._sortedBy = [...child._sortedBy];
          return newNode;
        }
      }
    }
    if (!newNode.groupBy || newNode.groupBy.length === 0) {
      newNode.physicalStrategy = PhysicalStrategy.UNGROUPED;
      newNode._sortedBy = [];
      return newNode;
    }
    if (this.canUsePerfectHashAggregate(newNode, child)) {
      newNode.physicalStrategy = PhysicalStrategy.PERFECT_HASH;
      newNode._sortedBy = [];
      return newNode;
    }
    newNode.physicalStrategy = PhysicalStrategy.HASH;
    newNode._sortedBy = [];
    return newNode;
  }
  rewriteSort(node) {
    const newNode = this.rewriteChildren(node);
    const childCard = newNode.children[0]._cardinality || DEFAULT_CARDINALITY;
    if (newNode.limit) {
      newNode._cardinality = Math.min(newNode.limit, childCard);
      newNode._cost = this.costModel.topNSortCost(childCard, newNode.limit);
    } else {
      newNode._cardinality = childCard;
      newNode._cost = this.costModel.sortCost(childCard);
    }
    newNode._sortedBy = newNode.orderKeys.map((o) => ({ key: this.getColumnKey(o.expr), direction: (o.direction || "ASC").toUpperCase() })).filter((e) => e.key);
    return newNode;
  }
  inferSortOrder(node) {
    if (node.type === PlanNodeType.SORT) {
      return node.orderKeys.map((o) => ({ key: this.getColumnKey(o.expr), direction: (o.direction || "ASC").toUpperCase() })).filter((e) => e.key);
    }
    if (node.type === PlanNodeType.INDEX_SCAN) {
      const key = `${(node.alias || node.table || "").toUpperCase()}.${(node.columnName || "").toUpperCase()}`;
      return [key];
    }
    if (node.type === PlanNodeType.FILTER || node.type === PlanNodeType.PROJECT || node.type === PlanNodeType.LIMIT) {
      if (node.children && node.children.length > 0) {
        return node.children[0]._sortedBy || [];
      }
    }
    return [];
  }
  getColumnKey(expr2) {
    if (!expr2) return null;
    if (expr2.kind === BoundExprKind.COLUMN_REF) {
      return `${expr2.tableAlias || ""}.${expr2.columnName}`.toUpperCase();
    }
    return null;
  }
  columnMatches(sortedKey, reqKey) {
    const s = sortedKey && typeof sortedKey === "object" ? sortedKey.key : sortedKey;
    const r = reqKey && typeof reqKey === "object" ? reqKey.key : reqKey;
    if (!s || !r) return false;
    if (s === r) return true;
    return s.split(".").pop() === r.split(".").pop();
  }
  isSortedBy(actualSortedKeys, requiredKeys) {
    if (!actualSortedKeys || actualSortedKeys.length === 0) return false;
    if (requiredKeys.length === 0) return false;
    for (let i = 0; i < requiredKeys.length; i++) {
      if (!this.columnMatches(actualSortedKeys[i], requiredKeys[i])) {
        return false;
      }
    }
    return true;
  }
  isSortedByPrefix(actualSortedKeys, requiredSet) {
    if (!actualSortedKeys || actualSortedKeys.length < requiredSet.length) return false;
    if (requiredSet.length === 0) return false;
    const prefix = actualSortedKeys.slice(0, requiredSet.length);
    for (const req of requiredSet) {
      if (!prefix.some((s) => this.columnMatches(s, req))) return false;
    }
    return true;
  }
  estimateNodeCardinality(node) {
    if (node.type === PlanNodeType.SCAN || node.type === PlanNodeType.INDEX_SCAN) {
      return this.cardEstimator.estimateScan(node.table);
    }
    if (node.type === PlanNodeType.FILTER) {
      const childCard = node.children[0]._cardinality || DEFAULT_CARDINALITY;
      return this.cardEstimator.estimateFilter(childCard, node.condition);
    }
    if (node.type === PlanNodeType.JOIN) {
      const leftCard = node.children[0]._cardinality || DEFAULT_CARDINALITY;
      const rightCard = node.children[1]._cardinality || DEFAULT_CARDINALITY;
      if (node.joinType === JoinType.SEMI) return this.cardEstimator.estimateSemiJoin(leftCard, rightCard, node.condition);
      if (node.joinType === JoinType.ANTI) return this.cardEstimator.estimateAntiJoin(leftCard, rightCard, node.condition);
      if (node.joinType === JoinType.MARK) return leftCard;
      if (node.joinType === JoinType.LEFT) {
        return this.cardEstimator.estimateLeftJoin ? this.cardEstimator.estimateLeftJoin(leftCard, rightCard, node.condition) : Math.max(leftCard, this.cardEstimator.estimateJoin(leftCard, rightCard, node.condition));
      }
      if (node.joinType === JoinType.CROSS) return leftCard * rightCard;
      return this.cardEstimator.estimateJoin(leftCard, rightCard, node.condition);
    }
    if (node.type === PlanNodeType.AGGREGATE) {
      const childCard = node.children[0]._cardinality || DEFAULT_CARDINALITY;
      return this.cardEstimator.estimateAggregate(childCard, node.groupBy?.length || 0, node.groupBy || []);
    }
    if (node.type === PlanNodeType.LIMIT) {
      const childCard = node.children[0]._cardinality || DEFAULT_CARDINALITY;
      return Math.min(node.count || childCard, childCard);
    }
    if (node.type === PlanNodeType.DISTINCT) {
      const childCard = node.children[0]._cardinality || DEFAULT_CARDINALITY;
      return Math.max(1, Math.round(Math.sqrt(childCard)));
    }
    if (node.children && node.children.length > 0) {
      return node.children[0]._cardinality || DEFAULT_CARDINALITY;
    }
    return 1e3;
  }
  extractEquiJoinKeys(condition) {
    const leftKeys = [];
    const rightKeys = [];
    const preds = this.splitAnd(condition);
    for (const pred of preds) {
      if (pred.kind === BoundExprKind.BINARY && pred.op === "=" && pred.left?.kind === BoundExprKind.COLUMN_REF && pred.right?.kind === BoundExprKind.COLUMN_REF) {
        leftKeys.push(this.getColumnKey(pred.left));
        rightKeys.push(this.getColumnKey(pred.right));
      }
    }
    return { leftKeys, rightKeys };
  }
  splitAnd(expr2) {
    if (!expr2) return [];
    if (expr2.kind === BoundExprKind.BINARY && expr2.op === "AND") {
      return [...this.splitAnd(expr2.left), ...this.splitAnd(expr2.right)];
    }
    return [expr2];
  }
  canUsePerfectHashAggregate(node, child) {
    if (!node.groupBy || node.groupBy.length === 0) return false;
    if (!node.groupBy.every((expr2) => expr2.kind === BoundExprKind.COLUMN_REF)) return false;
    const keyStats = node.groupBy.map((expr2) => this.cardEstimator.getColumnStats?.(expr2) || null);
    if (!keyStats.every(Boolean)) return false;
    let totalGroups = 1;
    for (const s of keyStats) {
      if (!s.ndv || s.ndv <= 0) return false;
      totalGroups *= s.ndv;
    }
    if (totalGroups > 256) return false;
    return keyStats.every((s) => this.hasCompactDomain(s));
  }
  hasCompactDomain(stats) {
    const ndv = stats.ndv || 0;
    if (ndv <= 0 || ndv > 256) return false;
    const min2 = toNumber2(stats.min);
    const max2 = toNumber2(stats.max);
    if (min2 !== null && max2 !== null && Number.isInteger(min2) && Number.isInteger(max2)) {
      const domainSize = max2 - min2 + 1;
      return domainSize > 0 && domainSize <= 4096;
    }
    return ndv <= 4;
  }
  estimateDownstreamSortSaving(originalNode, leftKeys, rightKeys, cardinality) {
    const parent = this.parentMap?.get(originalNode);
    if (!parent) return 0;
    const sortNode = this.findDownstreamSort(parent, originalNode);
    if (!sortNode || !sortNode.orderKeys) return 0;
    const sortKeys = sortNode.orderKeys.map((o) => this.getColumnKey(o.expr)).filter(Boolean);
    if (sortKeys.length === 0) return 0;
    const mergeOutputKeys = [...leftKeys, ...rightKeys];
    if (!this.isSortedByPrefix(mergeOutputKeys, sortKeys)) return 0;
    const card = cardinality || DEFAULT_CARDINALITY;
    return sortNode.limit ? this.costModel.topNSortCost(card, sortNode.limit) : this.costModel.sortCost(card);
  }
  findDownstreamSort(node, from) {
    if (!node) return null;
    if (node.type === PlanNodeType.SORT) return node;
    if (node.type === PlanNodeType.PROJECT || node.type === PlanNodeType.FILTER || node.type === PlanNodeType.LIMIT) {
      const parent = this.parentMap?.get(node);
      return parent ? this.findDownstreamSort(parent, node) : null;
    }
    return null;
  }
  isPureEquiJoin(condition) {
    if (!condition) return false;
    const preds = this.splitAnd(condition);
    return preds.length > 0 && preds.every(
      (pred) => pred.kind === BoundExprKind.BINARY && pred.op === "=" && pred.left?.kind === BoundExprKind.COLUMN_REF && pred.right?.kind === BoundExprKind.COLUMN_REF
    );
  }
};
function buildParentMap(root) {
  const map = /* @__PURE__ */ new Map();
  const queue = [root];
  while (queue.length > 0) {
    const node = queue.shift();
    if (node.children) {
      for (const child of node.children) {
        map.set(child, node);
        queue.push(child);
      }
    }
  }
  return map;
}
function toNumber2(value) {
  if (value === null || value === void 0) return null;
  if (typeof value === "bigint") return Number(value);
  if (typeof value === "number") return value;
  return null;
}

// src/execution/query-executor.js
init_buffer_shim();
init_logical_plan();

// src/execution/fragment-spec.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
init_data_type();

// src/execution/expression-eval.js
init_buffer_shim();
init_expression_binder();
init_data_type();
var LIKE_CACHE_MAX = 256;
function compileExpression(expr2, columnMapping) {
  if (!expr2) return () => null;
  switch (expr2.kind) {
    case BoundExprKind.COLUMN_REF: {
      const colIdx = resolveColumnIndex(expr2, columnMapping);
      return (chunk, rowIdx) => chunk.columns[colIdx]?.get(rowIdx) ?? null;
    }
    case BoundExprKind.LITERAL:
      return () => expr2.value;
    case BoundExprKind.BINARY: {
      const left = compileExpression(expr2.left, columnMapping);
      const right = compileExpression(expr2.right, columnMapping);
      return compileBinaryOp(expr2.op, left, right);
    }
    case BoundExprKind.UNARY: {
      const operand = compileExpression(expr2.operand, columnMapping);
      if (expr2.op === "-") return (c, r) => {
        const v = operand(c, r);
        return v == null ? null : -v;
      };
      if (expr2.op === "NOT") return (c, r) => {
        const v = operand(c, r);
        return v == null ? null : !v;
      };
      return operand;
    }
    case BoundExprKind.BETWEEN: {
      const e = compileExpression(expr2.expr, columnMapping);
      const lo = compileExpression(expr2.low, columnMapping);
      const hi = compileExpression(expr2.high, columnMapping);
      return (c, r) => {
        const v = e(c, r);
        if (v == null) return null;
        const loV = lo(c, r), hiV = hi(c, r);
        const ge = loV == null ? null : v >= loV;
        const le = hiV == null ? null : v <= hiV;
        let res;
        if (ge === false || le === false) res = false;
        else if (ge === null || le === null) res = null;
        else res = true;
        return expr2.negated ? res === null ? null : !res : res;
      };
    }
    case BoundExprKind.IN_LIST: {
      const e = compileExpression(expr2.expr, columnMapping);
      if (Array.isArray(expr2.list)) {
        const test = (v, has2) => {
          if (v == null) return null;
          const found = has2(v);
          return found ? true : found === null ? null : false;
        };
        if (expr2.list.every((i) => i.kind === BoundExprKind.LITERAL)) {
          const litHasNull = expr2.list.some((i) => i.value === null || i.value === void 0);
          const values = new Set(expr2.list.filter((i) => i.value != null).map((i) => normalizeComparable(i.value)));
          const has2 = (v) => values.has(normalizeComparable(v)) ? true : litHasNull ? null : false;
          return (c, r) => {
            const res = test(e(c, r), has2);
            return expr2.negated ? res === null ? null : !res : res;
          };
        }
        const items = expr2.list.map((i) => compileExpression(i, columnMapping));
        const has = (v, c, r) => {
          let anyNull = false;
          for (const i of items) {
            const iv = i(c, r);
            if (iv == null) {
              anyNull = true;
              continue;
            }
            if (iv == v) return true;
          }
          return anyNull ? null : false;
        };
        return (c, r) => {
          const v = e(c, r);
          if (v == null) return null;
          const found = has(v, c, r);
          const res = found ? true : found === null ? null : false;
          return expr2.negated ? res === null ? null : !res : res;
        };
      }
      return () => true;
    }
    case BoundExprKind.LIKE: {
      const e = compileExpression(expr2.expr, columnMapping);
      const p = compileExpression(expr2.pattern, columnMapping);
      const regexCache = /* @__PURE__ */ new Map();
      const regexKeys = [];
      return (c, r) => {
        const val = e(c, r);
        const pattern = p(c, r);
        if (val === null || val === void 0 || pattern === null || pattern === void 0) return null;
        const patternKey = String(pattern);
        let regex = regexCache.get(patternKey);
        if (!regex) {
          regex = likeToRegex(patternKey);
          if (regexCache.size >= LIKE_CACHE_MAX) {
            regexCache.delete(regexKeys.shift());
          }
          regexCache.set(patternKey, regex);
          regexKeys.push(patternKey);
        }
        const result = regex.test(String(val));
        return expr2.negated ? !result : result;
      };
    }
    case BoundExprKind.IS_NULL: {
      const e = compileExpression(expr2.expr, columnMapping);
      if (expr2.negated) return (c, r) => e(c, r) !== null && e(c, r) !== void 0;
      return (c, r) => e(c, r) === null || e(c, r) === void 0;
    }
    case BoundExprKind.CASE: {
      const whenClauses = expr2.whenClauses.map((wc) => ({
        cond: compileExpression(wc.condition, columnMapping),
        result: compileExpression(wc.result, columnMapping)
      }));
      const elseExpr = expr2.elseExpr ? compileExpression(expr2.elseExpr, columnMapping) : () => null;
      return (c, r) => {
        for (const wc of whenClauses) {
          if (wc.cond(c, r)) return wc.result(c, r);
        }
        return elseExpr(c, r);
      };
    }
    case BoundExprKind.CAST: {
      const e = compileExpression(expr2.expr, columnMapping);
      return (c, r) => castValue(e(c, r), expr2.targetType);
    }
    case BoundExprKind.EXTRACT: {
      const source = compileExpression(expr2.source, columnMapping);
      const srcType = expr2.source?.dataType || expr2.source?.resultType;
      return (c, r) => {
        const val = source(c, r);
        if (val === null) return null;
        if (srcType === DataType.TIMESTAMP || typeof val === "bigint" || val > 1e5) {
          const ts = epochMsToTimestamp(typeof val === "bigint" ? Number(val) : val);
          switch (expr2.field) {
            case "YEAR":
              return ts.year;
            case "MONTH":
              return ts.month;
            case "DAY":
              return ts.day;
            case "HOUR":
              return ts.hour;
            case "MINUTE":
              return ts.minute;
            case "SECOND":
              return ts.second;
            default:
              return null;
          }
        }
        const d = epochDaysToDate(val);
        switch (expr2.field) {
          case "YEAR":
            return d.year;
          case "MONTH":
            return d.month;
          case "DAY":
            return d.day;
          default:
            return null;
        }
      };
    }
    case BoundExprKind.FUNCTION: {
      const args = expr2.args.map((a) => compileExpression(a, columnMapping));
      return compileFunction(expr2.name, args);
    }
    case BoundExprKind.AGGREGATE: {
      const aggKey = aggExprKey(expr2);
      if (columnMapping && columnMapping.has(aggKey)) {
        const colIdx = columnMapping.get(aggKey);
        return (chunk, rowIdx) => chunk.columns[colIdx]?.get(rowIdx) ?? null;
      }
      return expr2.args.length > 0 ? compileExpression(expr2.args[0], columnMapping) : () => null;
    }
    case BoundExprKind.INTERVAL:
      return () => ({ value: expr2.value, unit: expr2.unit, _isInterval: true });
    case BoundExprKind.WINDOW: {
      const wKey = windowExprKey(expr2);
      if (columnMapping && columnMapping.has(wKey)) {
        const colIdx = columnMapping.get(wKey);
        return (chunk, rowIdx) => chunk.columns[colIdx]?.get(rowIdx) ?? null;
      }
      return () => null;
    }
    default:
      return () => null;
  }
}
function addInterval(epochDays, amount, unit) {
  if (unit === "DAY") return epochDays + amount;
  const d = epochDaysToDate(epochDays);
  if (unit === "YEAR") {
    return dateToEpochDays(d.year + amount, d.month, Math.min(d.day, daysInMonth(d.year + amount, d.month)));
  }
  if (unit === "MONTH") {
    let newMonth = d.month + amount;
    let newYear = d.year;
    while (newMonth > 12) {
      newMonth -= 12;
      newYear++;
    }
    while (newMonth < 1) {
      newMonth += 12;
      newYear--;
    }
    return dateToEpochDays(newYear, newMonth, Math.min(d.day, daysInMonth(newYear, newMonth)));
  }
  return epochDays + amount;
}
function daysInMonth(year, month) {
  return new Date(year, month, 0).getDate();
}
function toNum(val) {
  return typeof val === "bigint" ? Number(val) : val;
}
function normalizeComparable(val) {
  return typeof val === "bigint" ? Number(val) : val;
}
function numOp(a, b, fn) {
  return fn(toNum(a), toNum(b));
}
function compileBinaryOp(op, left, right) {
  switch (op) {
    case "=":
      return (c, r) => {
        const l = left(c, r), rv = right(c, r);
        return l == null || rv == null ? null : toNum(l) == toNum(rv);
      };
    case "<>":
      return (c, r) => {
        const l = left(c, r), rv = right(c, r);
        return l == null || rv == null ? null : toNum(l) != toNum(rv);
      };
    case "<":
      return (c, r) => {
        const l = left(c, r), rv = right(c, r);
        return l == null || rv == null ? null : toNum(l) < toNum(rv);
      };
    case ">":
      return (c, r) => {
        const l = left(c, r), rv = right(c, r);
        return l == null || rv == null ? null : toNum(l) > toNum(rv);
      };
    case "<=":
      return (c, r) => {
        const l = left(c, r), rv = right(c, r);
        return l == null || rv == null ? null : toNum(l) <= toNum(rv);
      };
    case ">=":
      return (c, r) => {
        const l = left(c, r), rv = right(c, r);
        return l == null || rv == null ? null : toNum(l) >= toNum(rv);
      };
    case "AND":
      return (c, r) => {
        const l = left(c, r), rv = right(c, r);
        if (l === false || rv === false) return false;
        if (l == null || rv == null) return null;
        return true;
      };
    case "OR":
      return (c, r) => {
        const l = left(c, r), rv = right(c, r);
        if (l === true || rv === true) return true;
        if (l == null || rv == null) return null;
        return false;
      };
    case "+":
      return (c, r) => {
        const l = left(c, r), rv = right(c, r);
        if (l === null || rv === null) return null;
        if (rv?._isInterval) return addInterval(toNum(l), rv.value, rv.unit);
        if (l?._isInterval) return addInterval(toNum(rv), l.value, l.unit);
        return numOp(l, rv, (a, b) => a + b);
      };
    case "-":
      return (c, r) => {
        const l = left(c, r), rv = right(c, r);
        if (l === null || rv === null) return null;
        if (rv?._isInterval) return addInterval(toNum(l), -rv.value, rv.unit);
        return numOp(l, rv, (a, b) => a - b);
      };
    case "*":
      return (c, r) => {
        const l = left(c, r), rv = right(c, r);
        return l !== null && rv !== null ? numOp(l, rv, (a, b) => a * b) : null;
      };
    case "/":
      return (c, r) => {
        const l = left(c, r), rv = right(c, r);
        return l !== null && rv !== null && rv !== 0 ? numOp(l, rv, (a, b) => a / b) : null;
      };
    case "%":
      return (c, r) => {
        const l = left(c, r), rv = right(c, r);
        return l !== null && rv !== null ? numOp(l, rv, (a, b) => a % b) : null;
      };
    case "||":
      return (c, r) => {
        const l = left(c, r), rv = right(c, r);
        return l !== null && rv !== null ? String(l) + String(rv) : null;
      };
    default:
      return () => null;
  }
}
function compileFunction(name, args) {
  switch (name.toUpperCase()) {
    case "SUBSTRING": {
      const [str, from, len] = args;
      return (c, r) => {
        const s = str(c, r);
        if (s === null) return null;
        const start = from(c, r) - 1;
        if (len) return String(s).substring(start, start + len(c, r));
        return String(s).substring(start);
      };
    }
    case "TRIM":
      return (c, r) => {
        const v = args[0](c, r);
        return v !== null ? String(v).trim() : null;
      };
    case "UPPER":
      return (c, r) => {
        const v = args[0](c, r);
        return v !== null ? String(v).toUpperCase() : null;
      };
    case "LOWER":
      return (c, r) => {
        const v = args[0](c, r);
        return v !== null ? String(v).toLowerCase() : null;
      };
    case "ABS":
      return (c, r) => {
        const v = args[0](c, r);
        return v !== null ? Math.abs(v) : null;
      };
    case "ROUND":
      return (c, r) => {
        const v = args[0](c, r);
        const d = args[1] ? args[1](c, r) : 0;
        if (v === null) return null;
        const factor = Math.pow(10, d);
        return Math.round(v * factor) / factor;
      };
    case "COALESCE":
      return (c, r) => {
        for (const a of args) {
          const v = a(c, r);
          if (v !== null && v !== void 0) return v;
        }
        return null;
      };
    case "NULLIF":
      return (c, r) => {
        const v1 = args[0](c, r), v2 = args[1](c, r);
        return v1 == v2 ? null : v1;
      };
    case "SQRT":
      return (c, r) => {
        const v = args[0](c, r);
        return v !== null ? Math.sqrt(v) : null;
      };
    case "LENGTH":
      return (c, r) => {
        const v = args[0](c, r);
        return v !== null ? String(v).length : null;
      };
    case "REPLACE":
      return (c, r) => {
        const s = args[0](c, r), from = args[1](c, r), to = args[2](c, r);
        if (s === null || from === null || to === null) return null;
        return String(s).split(String(from)).join(String(to));
      };
    default:
      return () => null;
  }
}
function resolveColumnIndex(expr2, columnMapping) {
  if (columnMapping) {
    const key = `${expr2.tableAlias}.${expr2.columnName}`.toUpperCase();
    if (columnMapping.has(key)) return columnMapping.get(key);
    const byName = `${expr2.columnName}`.toUpperCase();
    if (columnMapping.has(byName)) return columnMapping.get(byName);
  }
  return expr2.columnIndex;
}
function likeToRegex(pattern) {
  let regex = "^";
  for (const ch of pattern) {
    if (ch === "%") regex += ".*";
    else if (ch === "_") regex += ".";
    else if (".+*?^${}()|[]\\".includes(ch)) regex += "\\" + ch;
    else regex += ch;
  }
  regex += "$";
  return new RegExp(regex, "i");
}
function aggExprKey(expr2) {
  const name = expr2.name?.toUpperCase() || "AGG";
  const distinctTag = expr2.distinct ? "_DISTINCT" : "";
  if (expr2.args.length === 0) return `__AGG__${name}${distinctTag}`;
  const argKey = expr2.args.map((a) => {
    if (a.kind === BoundExprKind.COLUMN_REF) return `${a.tableAlias}.${a.columnName}`.toUpperCase();
    return JSON.stringify(a).slice(0, 30);
  }).join(",");
  return `__AGG__${name}${distinctTag}(${argKey})`;
}
function windowExprKey(expr2) {
  const name = expr2.name?.toUpperCase() || "WIN";
  const argKey = (expr2.args || []).map((a) => {
    if (a.kind === BoundExprKind.COLUMN_REF) return `${a.tableAlias}.${a.columnName}`.toUpperCase();
    return JSON.stringify(a).slice(0, 30);
  }).join(",");
  const partKey = (expr2.partitionBy || []).map((p) => {
    if (p.kind === BoundExprKind.COLUMN_REF) return `${p.tableAlias}.${p.columnName}`.toUpperCase();
    return "";
  }).join(",");
  return `__WIN__${name}(${argKey})[${partKey}]`;
}
function castValue(val, targetType) {
  if (val === null) return null;
  switch (targetType) {
    case DataType.INT32:
      return parseInt(val, 10) | 0;
    case DataType.INT64:
      return BigInt(parseInt(val, 10));
    case DataType.FLOAT64:
      return parseFloat(val);
    case DataType.VARCHAR: {
      if (typeof val === "bigint") return String(Number(val));
      return String(val);
    }
    case DataType.BOOLEAN:
      return !!val;
    case DataType.TIMESTAMP: {
      if (typeof val === "string") {
        return new Date(val).getTime();
      }
      return Number(val);
    }
    case DataType.DATE: {
      if (typeof val === "string") {
        const [y, m, d] = val.split("-").map(Number);
        return dateToEpochDays(y, m, d);
      }
      return val;
    }
    default:
      return val;
  }
}

// src/execution/operators/filter.js
init_buffer_shim();
init_chunk();
init_expression_binder();
init_data_type();
var OP_TO_FILTER = {
  "=": "filterEq",
  "<": "filterLt",
  ">": "filterGt",
  "<=": "filterLe",
  ">=": "filterGe"
};
var FilterOperator = class {
  constructor(predicate, evaluator, columnMapping, parallelDispatch) {
    this.predicate = predicate;
    this.evaluator = evaluator;
    this.columnMapping = columnMapping || null;
    this.parallelDispatch = parallelDispatch || null;
  }
  async init() {
  }
  async process(chunk) {
    const size = chunk.size;
    if (size === 0) return new DataChunk(chunk.columns, 0);
    if (this.parallelDispatch) {
      const plan = this._analyze(this.predicate);
      if (plan) {
        const result = await this._executeParallel(chunk, plan);
        if (result) return result;
      }
    }
    return this._executeFallback(chunk);
  }
  _analyze(expr2) {
    if (!expr2) return null;
    if (expr2.kind === BoundExprKind.COMPARISON || expr2.kind === BoundExprKind.BINARY) {
      const logical = this._analyzeLogical(expr2);
      if (logical) return logical;
      return this._analyzeComparison(expr2);
    }
    if (expr2.kind === BoundExprKind.BETWEEN) {
      return this._analyzeBetween(expr2);
    }
    return null;
  }
  _analyzeComparison(expr2) {
    const op = expr2.op;
    if (!OP_TO_FILTER[op]) return null;
    const { columnRef, literal } = this._extractColumnAndLiteral(expr2);
    if (!columnRef || literal === null) return null;
    const dataType = columnRef.dataType;
    if (!isFixedWidth(dataType)) return null;
    const operation = OP_TO_FILTER[op];
    if (!this.parallelDispatch.canParallelize(operation, dataType, 1)) return null;
    return {
      type: "simple",
      operation,
      dataType,
      columnIndex: columnRef.columnIndex,
      value: literal
    };
  }
  _analyzeBetween(expr2) {
    if (!expr2.expr || expr2.expr.kind !== BoundExprKind.COLUMN_REF) return null;
    const colRef = expr2.expr;
    if (!isFixedWidth(colRef.dataType)) return null;
    const low = this._extractLiteralValue(expr2.low);
    const high = this._extractLiteralValue(expr2.high);
    if (low === null || high === null) return null;
    return {
      type: "between",
      operation: "filterBetween",
      dataType: colRef.dataType,
      columnIndex: colRef.columnIndex,
      low,
      high
    };
  }
  _analyzeLogical(expr2) {
    if (expr2.op !== "AND" && expr2.op !== "OR") return null;
    const leftPlan = this._analyze(expr2.left);
    const rightPlan = this._analyze(expr2.right);
    if (!leftPlan || !rightPlan) return null;
    return {
      type: expr2.op === "AND" ? "and" : "or",
      left: leftPlan,
      right: rightPlan
    };
  }
  _extractColumnAndLiteral(expr2) {
    let columnRef = null;
    let literal = null;
    if (expr2.left?.kind === BoundExprKind.COLUMN_REF && expr2.right?.kind === BoundExprKind.LITERAL) {
      columnRef = expr2.left;
      literal = expr2.right.value;
    } else if (expr2.right?.kind === BoundExprKind.COLUMN_REF && expr2.left?.kind === BoundExprKind.LITERAL) {
      columnRef = expr2.right;
      literal = expr2.left.value;
    }
    return { columnRef, literal };
  }
  _extractLiteralValue(expr2) {
    if (!expr2 || expr2.kind !== BoundExprKind.LITERAL) return null;
    return expr2.value;
  }
  async _executeParallel(chunk, plan) {
    if (plan.type === "simple") return this._executeSimple(chunk, plan);
    if (plan.type === "between") return this._executeBetween(chunk, plan);
    if (plan.type === "and") return this._executeAnd(chunk, plan);
    if (plan.type === "or") return this._executeOr(chunk, plan);
    return null;
  }
  async _executeSimple(chunk, plan) {
    const column = chunk.columns[plan.columnIndex];
    const data = this._getColumnData(chunk, column);
    if (!data) return null;
    const result = await this.parallelDispatch.filterParallel(
      data,
      data.length,
      plan.operation,
      plan.dataType,
      { value: plan.value }
    );
    if (!result) return null;
    const count2 = this._dropNullRows(column, result.selectionVector, result.matchCount);
    return this._applySelectionVector(chunk, result.selectionVector, count2);
  }
  _dropNullRows(column, sv, count2) {
    if (!column.hasNulls) return count2;
    let w = 0;
    for (let i = 0; i < count2; i++) {
      if (!column.isNull(sv[i])) sv[w++] = sv[i];
    }
    return w;
  }
  async _executeBetween(chunk, plan) {
    const column = chunk.columns[plan.columnIndex];
    const data = this._getColumnData(chunk, column);
    if (!data) return null;
    const result = await this.parallelDispatch.filterParallel(
      data,
      data.length,
      plan.operation,
      plan.dataType,
      { low: plan.low, high: plan.high }
    );
    if (!result) return null;
    const count2 = this._dropNullRows(column, result.selectionVector, result.matchCount);
    return this._applySelectionVector(chunk, result.selectionVector, count2);
  }
  async _executeAnd(chunk, plan) {
    const leftResult = await this._executeParallel(chunk, plan.left);
    if (!leftResult || leftResult.size === 0) return new DataChunk(chunk.columns, 0);
    const rightResult = await this._executeParallel(chunk, plan.right);
    if (!rightResult || rightResult.size === 0) return new DataChunk(chunk.columns, 0);
    const leftSv = leftResult.selectionVector;
    const rightSv = rightResult.selectionVector;
    if (!leftSv || !rightSv) return null;
    const merged = intersectSorted(leftSv, leftResult.size, rightSv, rightResult.size);
    return this._applySelectionVector(chunk, merged.data, merged.count);
  }
  async _executeOr(chunk, plan) {
    const leftResult = await this._executeParallel(chunk, plan.left);
    const rightResult = await this._executeParallel(chunk, plan.right);
    if (!leftResult && !rightResult) return null;
    if (!leftResult) return rightResult;
    if (!rightResult) return leftResult;
    const leftSv = leftResult.selectionVector;
    const rightSv = rightResult.selectionVector;
    if (!leftSv && !rightSv) return null;
    if (!leftSv) return rightResult;
    if (!rightSv) return leftResult;
    const merged = unionSorted(leftSv, leftResult.size, rightSv, rightResult.size);
    return this._applySelectionVector(chunk, merged.data, merged.count);
  }
  _getColumnData(chunk, column) {
    if (chunk.selectionVector) return null;
    if (!column.data) return null;
    return column.data.subarray(0, column.length);
  }
  _applySelectionVector(chunk, sv, count2) {
    if (count2 === 0) return new DataChunk(chunk.columns, 0);
    if (count2 === chunk.size && !chunk.selectionVector) return chunk;
    const result = new DataChunk(chunk.columns, count2);
    result.setSelectionVector(sv.length === count2 ? sv : sv.subarray(0, count2), count2);
    return result;
  }
  _executeFallback(chunk) {
    const size = chunk.size;
    const sv = new Uint32Array(size);
    let count2 = 0;
    if (chunk.selectionVector) {
      const inputSv = chunk.selectionVector;
      for (let i = 0; i < size; i++) {
        const rowIdx = inputSv[i];
        if (this.evaluator(chunk, rowIdx)) {
          sv[count2++] = rowIdx;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        if (this.evaluator(chunk, i)) {
          sv[count2++] = i;
        }
      }
    }
    if (count2 === 0) return new DataChunk(chunk.columns, 0);
    if (count2 === size) return chunk;
    const result = new DataChunk(chunk.columns, count2);
    if (count2 > 64) {
      result.setSelectionVector(sv.subarray(0, count2), count2);
    } else {
      result.setSelectionVector(sv.slice(0, count2), count2);
    }
    return result;
  }
};
function intersectSorted(a, aLen, b, bLen) {
  const out = new Uint32Array(Math.min(aLen, bLen));
  let i = 0, j = 0, k = 0;
  while (i < aLen && j < bLen) {
    const va = a[i], vb = b[j];
    if (va === vb) {
      out[k++] = va;
      i++;
      j++;
    } else if (va < vb) {
      i++;
    } else {
      j++;
    }
  }
  return { data: out, count: k };
}
function unionSorted(a, aLen, b, bLen) {
  const out = new Uint32Array(aLen + bLen);
  let i = 0, j = 0, k = 0;
  while (i < aLen && j < bLen) {
    const va = a[i], vb = b[j];
    if (va === vb) {
      out[k++] = va;
      i++;
      j++;
    } else if (va < vb) {
      out[k++] = va;
      i++;
    } else {
      out[k++] = vb;
      j++;
    }
  }
  while (i < aLen) out[k++] = a[i++];
  while (j < bLen) out[k++] = b[j++];
  return { data: out, count: k };
}

// src/execution/operators/projection.js
init_buffer_shim();
init_column();
init_chunk();
init_expression_binder();

// src/execution/wasm-expr-eval.js
init_buffer_shim();
init_expression_binder();
init_dispatch();
init_config();
var NUMERIC_TYPES = /* @__PURE__ */ new Set(["INT32", "FLOAT64", "DATE", "DECIMAL"]);
var WIDEN_TYPES = /* @__PURE__ */ new Set(["INT32", "DATE"]);
var ARITH_OPS = /* @__PURE__ */ new Set(["+", "-", "*", "/"]);
var DECIMAL_SCALE = 100;
function isVectorizableExpr(expr2) {
  if (!expr2) return false;
  if (expr2.kind === BoundExprKind.COLUMN_REF) return NUMERIC_TYPES.has(expr2.dataType);
  if (expr2.kind === BoundExprKind.LITERAL) return typeof expr2.value === "number";
  if (expr2.kind === BoundExprKind.BINARY && ARITH_OPS.has(expr2.op)) {
    return isVectorizableExpr(expr2.left) && isVectorizableExpr(expr2.right);
  }
  if (expr2.kind === BoundExprKind.UNARY && expr2.op === "-") {
    return isVectorizableExpr(expr2.operand);
  }
  return false;
}
function extractColumnF64(col2, chunk, size) {
  const sv = chunk.selectionVector;
  if (col2.dataType === "DECIMAL") {
    const f64 = new Float64Array(size);
    if (sv) {
      for (let i = 0; i < size; i++) f64[i] = Number(col2.data[sv[i]]) / DECIMAL_SCALE;
    } else {
      for (let i = 0; i < size; i++) f64[i] = Number(col2.data[i]) / DECIMAL_SCALE;
    }
    return f64;
  }
  if (WIDEN_TYPES.has(col2.dataType)) {
    if (sv) {
      const compact = new Int32Array(size);
      for (let i = 0; i < size; i++) compact[i] = col2.data[sv[i]];
      return compact;
    }
    return col2.data.subarray(0, size);
  }
  if (col2.dataType === "FLOAT64") {
    if (sv) {
      const compact = new Float64Array(size);
      for (let i = 0; i < size; i++) compact[i] = col2.data[sv[i]];
      return compact;
    }
    return col2.data.subarray(0, size);
  }
  return null;
}
async function evalVectorized(expr2, chunk, columnMapping, size) {
  if (globalDispatch.kernels.size === 0) return null;
  if (expr2.kind === BoundExprKind.COLUMN_REF) {
    const idx = resolveColIndex(expr2, columnMapping);
    if (idx < 0) return null;
    const col2 = chunk.columns[idx];
    if (!col2 || !col2.data) return null;
    const raw = extractColumnF64(col2, chunk, size);
    if (!raw) return null;
    if (raw instanceof Int32Array) {
      const kernel = globalDispatch.lookup("widenI32ToF64", "INT32");
      return kernel ? await kernel(raw) : null;
    }
    return raw;
  }
  if (expr2.kind === BoundExprKind.LITERAL) {
    return Number(expr2.value);
  }
  if (expr2.kind === BoundExprKind.UNARY && expr2.op === "-") {
    const operand = await evalVectorized(expr2.operand, chunk, columnMapping, size);
    if (operand === null) return null;
    if (typeof operand === "number") return -operand;
    const kernel = globalDispatch.lookup("negF64", "FLOAT64");
    if (!kernel) return null;
    return await kernel(operand);
  }
  if (expr2.kind === BoundExprKind.BINARY && ARITH_OPS.has(expr2.op)) {
    const left = await evalVectorized(expr2.left, chunk, columnMapping, size);
    if (left === null) return null;
    const right = await evalVectorized(expr2.right, chunk, columnMapping, size);
    if (right === null) return null;
    const leftIsArr = left instanceof Float64Array || left instanceof Int32Array;
    const rightIsArr = right instanceof Float64Array || right instanceof Int32Array;
    if (leftIsArr && rightIsArr) {
      const name = { "+": "vecAddF64", "-": "vecSubF64", "*": "vecMulF64", "/": "vecDivF64" }[expr2.op];
      const kernel = globalDispatch.lookup(name, "FLOAT64");
      return kernel ? await kernel(left, right) : null;
    }
    if (leftIsArr && !rightIsArr) {
      const name = { "+": "scalarAddF64", "-": "scalarSubF64", "*": "scalarMulF64", "/": "scalarDivF64" }[expr2.op];
      const kernel = globalDispatch.lookup(name, "FLOAT64");
      return kernel ? await kernel(left, Number(right)) : null;
    }
    if (!leftIsArr && rightIsArr) {
      if (expr2.op === "+" || expr2.op === "*") {
        const name = expr2.op === "+" ? "scalarAddF64" : "scalarMulF64";
        const kernel = globalDispatch.lookup(name, "FLOAT64");
        return kernel ? await kernel(right, Number(left)) : null;
      }
      if (expr2.op === "-") {
        const kernel = globalDispatch.lookup("scalarSubRevF64", "FLOAT64");
        return kernel ? await kernel(Number(left), right) : null;
      }
      if (expr2.op === "/") {
        const kernel = globalDispatch.lookup("scalarDivRevF64", "FLOAT64");
        return kernel ? await kernel(Number(left), right) : null;
      }
    }
  }
  return null;
}
function resolveColIndex(expr2, columnMapping) {
  if (columnMapping) {
    const key = `${expr2.tableAlias}.${expr2.columnName}`.toUpperCase();
    if (columnMapping.has(key)) return columnMapping.get(key);
    const byName = expr2.columnName.toUpperCase();
    if (columnMapping.has(byName)) return columnMapping.get(byName);
  }
  return expr2.columnIndex >= 0 ? expr2.columnIndex : -1;
}

// src/execution/operators/projection.js
init_config();
init_bitmap();
function resolveColumnIndex2(expr2, columnMapping) {
  if (columnMapping) {
    const key = `${expr2.tableAlias}.${expr2.columnName}`.toUpperCase();
    if (columnMapping.has(key)) return columnMapping.get(key);
    const byName = `${expr2.columnName}`.toUpperCase();
    if (columnMapping.has(byName)) return columnMapping.get(byName);
  }
  return expr2.columnIndex;
}
function collectNullableColumns(expr2, chunk, columnMapping, acc) {
  if (!expr2 || typeof expr2 !== "object") return;
  if (expr2.kind === BoundExprKind.COLUMN_REF) {
    const col2 = chunk.columns[resolveColumnIndex2(expr2, columnMapping)];
    if (col2 && col2.hasNulls) acc.push(col2);
    return;
  }
  for (const key of ["left", "right", "operand", "expr"]) {
    if (expr2[key]) collectNullableColumns(expr2[key], chunk, columnMapping, acc);
  }
}
function applyNullMask(col2, expr2, chunk, columnMapping) {
  const nullable = [];
  collectNullableColumns(expr2, chunk, columnMapping, nullable);
  if (nullable.length === 0) return;
  const size = chunk.size;
  let hasNull = false;
  for (let i = 0; i < size; i++) {
    const row = chunk.activeRowIndex(i);
    let isNull = false;
    for (const src of nullable) {
      if (src.isNull(row)) {
        isNull = true;
        break;
      }
    }
    if (isNull) {
      clearBit(col2.nullBitmap, i);
      hasNull = true;
    } else setBit(col2.nullBitmap, i);
  }
  col2.hasNulls = hasNull;
}
async function tryWasmProject(expr2, chunk, columnMapping) {
  if (!isVectorizableExpr(expr2)) return null;
  if (chunk.size < Config.wasmMinChunkSize) return null;
  const result = await evalVectorized(expr2, chunk, columnMapping, chunk.size);
  if (result === null || typeof result === "number") return null;
  const col2 = new Column("FLOAT64", chunk.size);
  col2.data.set(result);
  col2.length = chunk.size;
  applyNullMask(col2, expr2, chunk, columnMapping);
  return col2;
}
var ProjectionOperator = class {
  constructor(expressions, evaluators, resultTypes = null, columnMapping = null, parallelDispatch) {
    this.expressions = expressions;
    this.evaluators = evaluators;
    this.resultTypes = resultTypes;
    this.columnMapping = columnMapping;
    this.parallelDispatch = parallelDispatch || null;
    this.colRefIndices = expressions.map((expr2) => {
      if (expr2?.kind === BoundExprKind.COLUMN_REF) {
        return this._resolveColIdx(expr2);
      }
      return -1;
    });
  }
  async init() {
  }
  async process(chunk) {
    if (chunk.size === 0) return new DataChunk([], 0);
    const outputCols = [];
    const needsFlatten = !!chunk.selectionVector;
    for (let e = 0; e < this.evaluators.length; e++) {
      const colRefIdx = this.colRefIndices[e];
      const dataType = this.resultTypes ? this.resultTypes[e] : this.expressions[e]?.dataType || this.expressions[e]?.resultType || "VARCHAR";
      if (colRefIdx >= 0 && !needsFlatten) {
        const srcCol = chunk.columns[colRefIdx];
        if (srcCol) {
          outputCols.push(srcCol);
          continue;
        }
      }
      if (this.expressions[e] && !this.parallelDispatch) {
        const wasmCol = await tryWasmProject(this.expressions[e], chunk, this.columnMapping);
        if (wasmCol) {
          outputCols.push(wasmCol);
          continue;
        }
      }
      const evalFn = this.evaluators[e];
      const col2 = new Column(dataType, chunk.size || 1);
      for (let i = 0; i < chunk.size; i++) {
        const rowIdx = chunk.activeRowIndex(i);
        const val = evalFn(chunk, rowIdx);
        col2.set(i, typeof val === "bigint" && dataType !== "INT64" ? Number(val) : val);
      }
      col2.length = chunk.size;
      outputCols.push(col2);
    }
    return new DataChunk(outputCols, chunk.size);
  }
  _resolveColIdx(expr2) {
    if (this.columnMapping) {
      const key = `${expr2.tableAlias}.${expr2.columnName}`.toUpperCase();
      if (this.columnMapping.has(key)) return this.columnMapping.get(key);
      const byName = expr2.columnName.toUpperCase();
      if (this.columnMapping.has(byName)) return this.columnMapping.get(byName);
    }
    return expr2.columnIndex >= 0 ? expr2.columnIndex : -1;
  }
};

// src/execution/operators/hash-aggregate.js
init_buffer_shim();
init_column();
init_chunk();
init_data_type();
init_dispatch();

// src/utils/hash.js
init_buffer_shim();
var scratchF64 = new Float64Array(1);
var scratchU32 = new Uint32Array(scratchF64.buffer);
function hashValue(value) {
  if (typeof value === "number") {
    scratchF64[0] = value;
    let h = scratchU32[0] ^ Math.imul(scratchU32[1], 2654435761);
    h = Math.imul(h ^ h >>> 16, 2246822507);
    return (h ^ h >>> 13) >>> 0;
  }
  if (typeof value === "string") {
    let h = 2166136261;
    for (let i = 0; i < value.length; i++) {
      h ^= value.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }
  if (typeof value === "bigint") {
    return hashValue(Number(BigInt.asIntN(53, value)));
  }
  if (typeof value === "boolean") {
    return value ? 1 : 0;
  }
  return 0;
}

// src/execution/operators/hash-aggregate.js
var HashAggregateOperator = class {
  constructor(groupByExtractors, groupByTypes, aggregateDefs) {
    this.groupByExtractors = groupByExtractors;
    this.groupByTypes = groupByTypes;
    this.aggregateDefs = aggregateDefs;
    this.hasCachedValues = aggregateDefs.some((def) => def.valueKey);
    this.groups = /* @__PURE__ */ new Map();
    this.groupKeys = [];
  }
  async init() {
  }
  async consume(chunk) {
    if (this.groupByExtractors.length === 0 && chunk.size > 0) {
      if (globalDispatch && globalDispatch.kernels.size > 0) {
        const wasmHandled = await this._tryWasmUngrouped(chunk);
        if (wasmHandled) return;
      }
    }
    const size = chunk.size;
    const hasSv = !!chunk.selectionVector;
    const sv = chunk.selectionVector;
    const groupByCount = this.groupByExtractors.length;
    const aggCount = this.aggregateDefs.length;
    const groupByVals = new Array(groupByCount);
    for (let g = 0; g < groupByCount; g++) {
      groupByVals[g] = new Array(size);
      const fn = this.groupByExtractors[g];
      for (let i = 0; i < size; i++) {
        const rowIdx = hasSv ? sv[i] : i;
        groupByVals[g][i] = fn(chunk, rowIdx);
      }
    }
    const aggVals = new Array(aggCount);
    const extractedKeys = /* @__PURE__ */ new Set();
    for (let a = 0; a < aggCount; a++) {
      const def = this.aggregateDefs[a];
      if (this.hasCachedValues && def.valueKey && extractedKeys.has(def.valueKey)) {
        for (let prev = 0; prev < a; prev++) {
          if (this.aggregateDefs[prev].valueKey === def.valueKey) {
            aggVals[a] = aggVals[prev];
            break;
          }
        }
      } else {
        aggVals[a] = new Array(size);
        for (let i = 0; i < size; i++) {
          const rowIdx = hasSv ? sv[i] : i;
          aggVals[a][i] = def.extractValue(chunk, rowIdx);
        }
        if (def.valueKey) extractedKeys.add(def.valueKey);
      }
    }
    for (let i = 0; i < size; i++) {
      let key;
      if (groupByCount === 0) {
        key = GLOBAL_GROUP_KEY;
      } else if (groupByCount === 1) {
        key = groupByVals[0][i];
      } else {
        const parts = new Array(groupByCount);
        for (let g = 0; g < groupByCount; g++) {
          const v = groupByVals[g][i];
          parts[g] = typeof v === "bigint" ? v.toString() : String(v);
        }
        key = parts.join("|");
      }
      let group = this.groups.get(key);
      if (!group) {
        const gv = new Array(groupByCount);
        for (let g = 0; g < groupByCount; g++) gv[g] = groupByVals[g][i];
        group = {
          groupValues: gv,
          accumulators: this.aggregateDefs.map((def) => def.createAccumulator())
        };
        this.groups.set(key, group);
      }
      for (let a = 0; a < aggCount; a++) {
        group.accumulators[a].add(aggVals[a][i]);
      }
    }
  }
  async finalize() {
    const groupCount = this.groups.size;
    if (groupCount === 0) {
      if (this.groupByExtractors.length === 0) {
        const cols = this.aggregateDefs.map((def) => {
          const col2 = new Column(def.resultType, 1);
          const acc = def.createAccumulator();
          col2.set(0, acc.result());
          col2.length = 1;
          return col2;
        });
        return [new DataChunk(cols, 1)];
      }
      return [];
    }
    const groupByCount = this.groupByExtractors.length;
    const aggCount = this.aggregateDefs.length;
    const totalCols = groupByCount + aggCount;
    const chunks = [];
    const allGroups = Array.from(this.groups.values());
    for (let start = 0; start < allGroups.length; start += DEFAULT_CHUNK_SIZE) {
      const end = Math.min(start + DEFAULT_CHUNK_SIZE, allGroups.length);
      const batchSize = end - start;
      const columns = new Array(totalCols);
      for (let g = 0; g < groupByCount; g++) {
        columns[g] = new Column(this.groupByTypes[g] || DataType.VARCHAR, batchSize);
      }
      for (let a = 0; a < aggCount; a++) {
        columns[groupByCount + a] = new Column(this.aggregateDefs[a].resultType, batchSize);
      }
      for (let r = 0; r < batchSize; r++) {
        const group = allGroups[start + r];
        for (let g = 0; g < groupByCount; g++) {
          const val = group.groupValues[g];
          columns[g].set(r, typeof val === "bigint" ? Number(val) : val);
        }
        for (let a = 0; a < aggCount; a++) {
          columns[groupByCount + a].set(r, group.accumulators[a].result());
        }
      }
      for (const col2 of columns) col2.length = batchSize;
      chunks.push(new DataChunk(columns, batchSize));
    }
    return chunks;
  }
  exportPartials(partitionCount) {
    const mask = partitionCount - 1;
    const partitions = Array.from({ length: partitionCount }, () => []);
    for (const [key, group] of this.groups) {
      partitions[hashGroupKey(key) & mask].push({
        key,
        groupValues: group.groupValues,
        states: group.accumulators.map((acc) => acc.exportState())
      });
    }
    return partitions;
  }
  absorbPartials(partials) {
    const aggCount = this.aggregateDefs.length;
    for (const partial of partials) {
      let group = this.groups.get(partial.key);
      if (!group) {
        group = {
          groupValues: partial.groupValues,
          accumulators: this.aggregateDefs.map((def) => def.createAccumulator())
        };
        this.groups.set(partial.key, group);
      }
      for (let a = 0; a < aggCount; a++) {
        group.accumulators[a].mergeState(partial.states[a]);
      }
    }
  }
  _resolveWasmAggKernel(def) {
    const name = def.name?.toUpperCase();
    if (!name) return null;
    if (name === "SUM" && def.resultType === "FLOAT64") {
      if (globalDispatch.has("sumF64", "FLOAT64")) return { kernelKey: "sumF64", dataType: "FLOAT64", kind: "SUM" };
      if (globalDispatch.has("sumI32", "INT32")) return { kernelKey: "sumI32", dataType: "INT32", kind: "SUM" };
    }
    if (name === "MIN") {
      if (def.resultType === "FLOAT64" && globalDispatch.has("minF64", "FLOAT64")) return { kernelKey: "minF64", dataType: "FLOAT64", kind: "MIN" };
      if (def.resultType === "INT32" && globalDispatch.has("minI32", "INT32")) return { kernelKey: "minI32", dataType: "INT32", kind: "MIN" };
    }
    if (name === "MAX") {
      if (def.resultType === "FLOAT64" && globalDispatch.has("maxF64", "FLOAT64")) return { kernelKey: "maxF64", dataType: "FLOAT64", kind: "MAX" };
      if (def.resultType === "INT32" && globalDispatch.has("maxI32", "INT32")) return { kernelKey: "maxI32", dataType: "INT32", kind: "MAX" };
    }
    if (name === "COUNT") {
      return { kernelKey: "countBits", dataType: "UINT8", kind: "COUNT" };
    }
    if (name === "COUNT_STAR") {
      return { kernelKey: null, dataType: null, kind: "COUNT_STAR" };
    }
    if (name === "AVG" && def.resultType === "FLOAT64") {
      if (globalDispatch.has("sumF64", "FLOAT64")) return { kernelKey: "sumF64", dataType: "FLOAT64", kind: "AVG" };
    }
    return null;
  }
  async _tryWasmUngrouped(chunk) {
    const size = chunk.size;
    const contributions = new Array(this.aggregateDefs.length);
    for (let a = 0; a < this.aggregateDefs.length; a++) {
      const def = this.aggregateDefs[a];
      const resolved = this._resolveWasmAggKernel(def);
      if (!resolved) return false;
      if (resolved.kind === "COUNT_STAR") {
        contributions[a] = { kind: "count", n: size };
        continue;
      }
      if (resolved.kind === "COUNT") {
        if (def._wasmColIndex === void 0 || def._wasmColIndex === null) return false;
        const column2 = chunk.columns[def._wasmColIndex];
        if (!column2) return false;
        if (!column2.hasNulls) {
          contributions[a] = { kind: "count", n: size };
        } else {
          const kernel2 = globalDispatch.lookup("countBits", "UINT8");
          if (!kernel2) return false;
          contributions[a] = { kind: "count", n: await kernel2(column2.nullBitmap, size) };
        }
        continue;
      }
      if (def._wasmColIndex === void 0 || def._wasmColIndex === null) return false;
      const column = chunk.columns[def._wasmColIndex];
      if (!column || !column.data || column.hasNulls) return false;
      const colType = column.dataType;
      const matches = resolved.dataType === "FLOAT64" && colType === "FLOAT64" || resolved.dataType === "INT32" && (colType === "INT32" || colType === "DATE");
      if (!matches) return false;
      const kernel = globalDispatch.lookup(resolved.kernelKey, resolved.dataType);
      if (!kernel) return false;
      const result = await kernel(column.data.subarray(0, size));
      contributions[a] = resolved.kind === "AVG" ? { kind: "avg", sum: result, n: size } : { kind: "value", result };
    }
    let group = this.groups.get(GLOBAL_GROUP_KEY);
    if (!group) {
      group = {
        groupValues: [],
        accumulators: this.aggregateDefs.map((def) => def.createAccumulator())
      };
      this.groups.set(GLOBAL_GROUP_KEY, group);
    }
    for (let a = 0; a < contributions.length; a++) {
      const c = contributions[a];
      const acc = group.accumulators[a];
      if (c.kind === "count") acc.count += c.n;
      else if (c.kind === "avg") {
        acc.sum += c.sum;
        acc.count += c.n;
      } else acc.add(c.result);
    }
    return true;
  }
};
var GLOBAL_GROUP_KEY = "__ALL__";
function hashGroupKey(key) {
  if (key === null || key === void 0) return 0;
  return hashValue(key);
}
var SumAccumulator = class {
  constructor() {
    this.sum = 0;
    this.hasValue = false;
  }
  add(val) {
    if (val !== null && val !== void 0) {
      this.sum += typeof val === "bigint" ? Number(val) : Number(val);
      this.hasValue = true;
    }
  }
  result() {
    return this.hasValue ? this.sum : null;
  }
  exportState() {
    return this.hasValue ? this.sum : null;
  }
  mergeState(state) {
    if (state !== null && state !== void 0) {
      this.sum += state;
      this.hasValue = true;
    }
  }
};
var CountAccumulator = class {
  constructor() {
    this.count = 0;
  }
  add(val) {
    if (val !== null && val !== void 0) this.count++;
  }
  result() {
    return this.count;
  }
  exportState() {
    return this.count;
  }
  mergeState(state) {
    this.count += state;
  }
};
var CountStarAccumulator = class {
  constructor() {
    this.count = 0;
  }
  add() {
    this.count++;
  }
  result() {
    return this.count;
  }
  exportState() {
    return this.count;
  }
  mergeState(state) {
    this.count += state;
  }
};
var AvgAccumulator = class {
  constructor() {
    this.sum = 0;
    this.count = 0;
  }
  add(val) {
    if (val !== null && val !== void 0) {
      this.sum += Number(val);
      this.count++;
    }
  }
  result() {
    return this.count > 0 ? this.sum / this.count : null;
  }
  exportState() {
    return { sum: this.sum, count: this.count };
  }
  mergeState(state) {
    this.sum += state.sum;
    this.count += state.count;
  }
};
var AvgFinalAccumulator = class {
  constructor() {
    this.sum = 0;
    this.count = 0;
  }
  add(pair) {
    if (!pair) return;
    const s = pair[0], c = pair[1];
    if (s !== null && s !== void 0 && c !== null && c !== void 0) {
      this.sum += Number(s);
      this.count += Number(c);
    }
  }
  result() {
    return this.count > 0 ? this.sum / this.count : null;
  }
  exportState() {
    return { sum: this.sum, count: this.count };
  }
  mergeState(state) {
    this.sum += state.sum;
    this.count += state.count;
  }
};
var MinAccumulator = class {
  constructor() {
    this.min = null;
  }
  add(val) {
    if (val !== null && val !== void 0 && (this.min === null || val < this.min)) this.min = val;
  }
  result() {
    return this.min;
  }
  exportState() {
    return this.min;
  }
  mergeState(state) {
    if (state !== null && state !== void 0 && (this.min === null || state < this.min)) this.min = state;
  }
};
var MaxAccumulator = class {
  constructor() {
    this.max = null;
  }
  add(val) {
    if (val !== null && val !== void 0 && (this.max === null || val > this.max)) this.max = val;
  }
  result() {
    return this.max;
  }
  exportState() {
    return this.max;
  }
  mergeState(state) {
    if (state !== null && state !== void 0 && (this.max === null || state > this.max)) this.max = state;
  }
};
var CountDistinctAccumulator = class {
  constructor() {
    this.values = /* @__PURE__ */ new Set();
  }
  add(val) {
    if (val !== null && val !== void 0) this.values.add(typeof val === "bigint" ? Number(val) : val);
  }
  result() {
    return this.values.size;
  }
  exportState() {
    return Array.from(this.values);
  }
  mergeState(state) {
    for (const val of state) this.values.add(val);
  }
};
function getAccumulatorFactory(name, distinct = false) {
  if (distinct && name.toUpperCase() === "COUNT") {
    return () => new CountDistinctAccumulator();
  }
  switch (name.toUpperCase()) {
    case "SUM":
      return () => new SumAccumulator();
    case "COUNT":
      return () => new CountAccumulator();
    case "COUNT_STAR":
      return () => new CountStarAccumulator();
    case "AVG":
      return () => new AvgAccumulator();
    case "AVG_PARTIAL":
      return () => new AvgAccumulator();
    case "AVG_FINAL":
      return () => new AvgFinalAccumulator();
    case "MIN":
      return () => new MinAccumulator();
    case "MAX":
      return () => new MaxAccumulator();
    default:
      throw new Error(`Unknown aggregate: ${name}`);
  }
}

// src/execution/fragment-spec.js
var StageKind = {
  FILTER: "filter",
  PROJECT: "project"
};
function normalizeExecType(dt) {
  if (dt === DataType.DECIMAL || dt === DataType.INT64) return DataType.FLOAT64;
  return dt;
}
function normalizeAggResultType(agg) {
  const name = (agg.name || "").toUpperCase();
  if (name === "COUNT" || name === "COUNT_STAR") return DataType.INT32;
  return DataType.FLOAT64;
}
function expressionCacheKey(expr2) {
  if (!expr2 || typeof expr2 !== "object") return String(expr2);
  switch (expr2.kind) {
    case BoundExprKind.COLUMN_REF:
      return `COL:${expr2.tableAlias || ""}.${expr2.columnName}`;
    case BoundExprKind.LITERAL:
      return `LIT:${String(expr2.value)}`;
    case BoundExprKind.BINARY:
      return `BIN:${expr2.op}:${expressionCacheKey(expr2.left)}:${expressionCacheKey(expr2.right)}`;
    case BoundExprKind.UNARY:
      return `UNARY:${expr2.op}:${expressionCacheKey(expr2.operand)}`;
    case BoundExprKind.CASE:
      return `CASE:${JSON.stringify(expr2)}`;
    default:
      return JSON.stringify(expr2);
  }
}
function schemaMappingOf(schema) {
  const mapping = /* @__PURE__ */ new Map();
  for (let i = 0; i < schema.length; i++) {
    const col2 = schema[i];
    const key = `${col2.tableAlias || ""}.${col2.name}`.toUpperCase();
    mapping.set(key, i);
    if (!mapping.has(col2.name.toUpperCase())) {
      mapping.set(col2.name.toUpperCase(), i);
    }
  }
  return mapping;
}
function projectionSchemaOf(expressions) {
  return expressions.map((expr2, i) => ({
    name: expr2?.outputName || expr2?.alias || expr2?.name || expr2?.columnName || `col${i}`,
    dataType: normalizeExecType(expr2?.dataType || expr2?.resultType || DataType.VARCHAR),
    tableAlias: ""
  }));
}
function collectColumnRefs(node, acc = []) {
  if (!node || typeof node !== "object") return acc;
  if (Array.isArray(node)) {
    for (const item of node) collectColumnRefs(item, acc);
    return acc;
  }
  if (node.kind === BoundExprKind.COLUMN_REF) {
    acc.push(node);
    return acc;
  }
  for (const value of Object.values(node)) collectColumnRefs(value, acc);
  return acc;
}
function resolveRef(ref, mapping) {
  const qualified = `${ref.tableAlias || ""}.${ref.columnName}`.toUpperCase();
  if (mapping.has(qualified)) return mapping.get(qualified);
  const bare = String(ref.columnName).toUpperCase();
  if (mapping.has(bare)) return mapping.get(bare);
  return void 0;
}
function exprResolvable(exprOrList, mapping) {
  for (const ref of collectColumnRefs(exprOrList)) {
    if (resolveRef(ref, mapping) === void 0) return false;
  }
  return true;
}
function buildAggregateDefs(aggregates, columnMapping) {
  const valueKeyCounts = /* @__PURE__ */ new Map();
  for (const agg of aggregates) {
    if (!agg.args || agg.args.length === 0) continue;
    const key = expressionCacheKey(agg.args[0]);
    valueKeyCounts.set(key, (valueKeyCounts.get(key) || 0) + 1);
  }
  return aggregates.map((agg) => {
    const hasArgs = agg.args && agg.args.length > 0;
    const valueExtractor = hasArgs ? compileExpression(agg.args[0], columnMapping) : () => 1;
    const valueKey = hasArgs ? expressionCacheKey(agg.args[0]) : null;
    let wasmColIndex;
    if (hasArgs && agg.args[0].kind === BoundExprKind.COLUMN_REF) {
      const resolved = resolveRef(agg.args[0], columnMapping);
      if (resolved !== void 0) wasmColIndex = resolved;
    }
    return {
      name: agg.name,
      valueKey: valueKey && valueKeyCounts.get(valueKey) > 1 ? valueKey : null,
      resultType: normalizeAggResultType(agg),
      createAccumulator: getAccumulatorFactory(agg.name, agg.distinct),
      extractValue: (chunk, rowIdx) => {
        const val = valueExtractor(chunk, rowIdx);
        return typeof val === "bigint" ? Number(val) : val;
      },
      _wasmColIndex: wasmColIndex,
      _sourceExpr: hasArgs ? agg.args[0] : null,
      _columnMapping: columnMapping
    };
  });
}
function extractStageChain(startNode) {
  const stages = [];
  let current = startNode;
  while (current) {
    if (current.type === PlanNodeType.FILTER) {
      stages.push({ kind: StageKind.FILTER, condition: current.condition });
      current = current.children[0];
    } else if (current.type === PlanNodeType.PROJECT) {
      stages.push({ kind: StageKind.PROJECT, expressions: current.expressions });
      current = current.children[0];
    } else if (current.type === PlanNodeType.SCAN) {
      stages.reverse();
      return {
        table: current.table,
        alias: current.alias || current.table,
        scanColumns: current.columns,
        stages
      };
    } else {
      return null;
    }
  }
  return null;
}
function extractAggregateFragment(node) {
  return extractStageChain(node.children[0]);
}
function stagedSchemaOf(baseSchema, stages) {
  let schema = baseSchema;
  for (const stage of stages) {
    if (stage.kind === StageKind.PROJECT) {
      schema = projectionSchemaOf(stage.expressions);
    }
  }
  return schema;
}
function stagesResolvable(baseSchema, stages) {
  let schema = baseSchema;
  let mapping = schemaMappingOf(schema);
  for (const stage of stages) {
    if (stage.kind === StageKind.FILTER) {
      if (!exprResolvable(stage.condition, mapping)) return false;
    } else {
      if (!exprResolvable(stage.expressions, mapping)) return false;
      schema = projectionSchemaOf(stage.expressions);
      mapping = schemaMappingOf(schema);
    }
  }
  return true;
}
function buildFragmentSpec(fragment, node, storageSchema) {
  const aliased = storageSchema.map((col2) => ({
    name: col2.name,
    dataType: col2.dataType,
    tableAlias: fragment.alias
  }));
  const baseRefs = [];
  let prunedAtProject = false;
  for (const stage of fragment.stages) {
    if (stage.kind === StageKind.FILTER) {
      collectColumnRefs(stage.condition, baseRefs);
    } else {
      collectColumnRefs(stage.expressions, baseRefs);
      prunedAtProject = true;
      break;
    }
  }
  if (!prunedAtProject) {
    collectColumnRefs(node.groupBy || [], baseRefs);
    for (const agg of node.aggregates) collectColumnRefs(agg.args || [], baseRefs);
  }
  const fullMapping = schemaMappingOf(aliased);
  const needed = /* @__PURE__ */ new Set();
  for (const ref of baseRefs) {
    const idx = resolveRef(ref, fullMapping);
    if (idx === void 0) return null;
    needed.add(idx);
  }
  const columnIndexes = Array.from(needed).sort((a, b) => a - b);
  const baseSchema = columnIndexes.map((i) => aliased[i]);
  const spec = {
    baseSchema,
    stages: fragment.stages,
    groupBy: node.groupBy || [],
    aggregates: node.aggregates.map((agg) => ({
      name: agg.name,
      distinct: !!agg.distinct,
      args: agg.args || []
    }))
  };
  if (!validateFragmentSpec(spec)) return null;
  let estimatedRowBytes = Math.ceil(columnIndexes.length / 8);
  for (const i of columnIndexes) {
    estimatedRowBytes += byteWidthFor(aliased[i].dataType) || Uint16Array.BYTES_PER_ELEMENT;
  }
  return { spec, columnIndexes, estimatedRowBytes };
}
function validateFragmentSpec(spec) {
  if (!stagesResolvable(spec.baseSchema, spec.stages)) return false;
  const mapping = schemaMappingOf(stagedSchemaOf(spec.baseSchema, spec.stages));
  for (const groupExpr of spec.groupBy) {
    if (!exprResolvable(groupExpr, mapping)) return false;
  }
  for (const agg of spec.aggregates) {
    if (!exprResolvable(agg.args, mapping)) return false;
  }
  return true;
}
function execResolvedIndex(ref, mapping) {
  const qualified = `${ref.tableAlias}.${ref.columnName}`.toUpperCase();
  if (mapping.has(qualified)) return mapping.get(qualified);
  const bare = `${ref.columnName}`.toUpperCase();
  if (mapping.has(bare)) return mapping.get(bare);
  return ref.columnIndex;
}
function refsResolveIdentically(exprOrList, mainMapping, workerMapping) {
  for (const ref of collectColumnRefs(exprOrList)) {
    const mainIdx = execResolvedIndex(ref, mainMapping);
    const workerIdx = execResolvedIndex(ref, workerMapping);
    if (mainIdx === void 0 || mainIdx === null || mainIdx !== workerIdx) return false;
  }
  return true;
}
function plainSchemaOf(schema) {
  return schema.map((col2) => ({ name: col2.name, dataType: col2.dataType, tableAlias: col2.tableAlias || "" }));
}
function schemasEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].name.toUpperCase() !== b[i].name.toUpperCase()) return false;
    if (a[i].dataType !== b[i].dataType) return false;
    if ((a[i].tableAlias || "").toUpperCase() !== (b[i].tableAlias || "").toUpperCase()) return false;
  }
  return true;
}
function buildJoinSpec({
  build,
  probe,
  buildKeys,
  probeKeys,
  residualCondition,
  joinType,
  buildPreserved,
  uniqueKeys,
  buildMapping,
  probeMapping,
  combinedMapping
}) {
  if (!stagesResolvable(build.baseSchema, build.stages)) return null;
  if (!stagesResolvable(probe.baseSchema, probe.stages)) return null;
  const workerBuildMapping = schemaMappingOf(stagedSchemaOf(build.baseSchema, build.stages));
  const workerProbeMapping = schemaMappingOf(stagedSchemaOf(probe.baseSchema, probe.stages));
  const workerCombinedMapping = schemaMappingOf([...build.schema, ...probe.schema]);
  if (!refsResolveIdentically(buildKeys, buildMapping, workerBuildMapping)) return null;
  if (!refsResolveIdentically(probeKeys, probeMapping, workerProbeMapping)) return null;
  if (residualCondition && !refsResolveIdentically(residualCondition, combinedMapping, workerCombinedMapping)) return null;
  return {
    build,
    probe,
    buildKeys,
    probeKeys,
    residualCondition: residualCondition || null,
    joinType,
    buildPreserved: !!buildPreserved,
    uniqueKeys: !!uniqueKeys,
    buildColCount: build.schema.length,
    probeColCount: probe.schema.length
  };
}

// src/execution/result-sink.js
init_buffer_shim();
init_config();
var ResultSink = class {
  constructor(streaming = false) {
    this._streaming = streaming;
    this._capacity = Config.sinkQueueCapacity;
    this._queue = new Array(this._capacity);
    this._head = 0;
    this._tail = 0;
    this._count = 0;
    this._totalRows = 0;
    this._done = false;
    this._error = null;
    this._producerResolve = null;
    this._consumerResolve = null;
    this._collected = [];
  }
  async init() {
  }
  async consume(chunk) {
    if (!chunk || chunk.size === 0) return;
    if (this._error) throw this._error;
    this._totalRows += chunk.size;
    if (!this._streaming) {
      this._collected.push(chunk);
      return;
    }
    if (this._count === this._capacity) {
      await new Promise((resolve) => {
        this._producerResolve = resolve;
      });
    }
    if (this._error) throw this._error;
    this._queue[this._tail] = chunk;
    this._tail = (this._tail + 1) % this._capacity;
    this._count++;
    if (this._consumerResolve) {
      const resolve = this._consumerResolve;
      this._consumerResolve = null;
      resolve();
    }
  }
  async finalize() {
    this._done = true;
    if (this._consumerResolve) {
      const resolve = this._consumerResolve;
      this._consumerResolve = null;
      resolve();
    }
  }
  error(err) {
    this._error = err;
    this._done = true;
    if (this._consumerResolve) {
      const resolve = this._consumerResolve;
      this._consumerResolve = null;
      resolve();
    }
    if (this._producerResolve) {
      const resolve = this._producerResolve;
      this._producerResolve = null;
      resolve();
    }
  }
  get totalRows() {
    return this._totalRows;
  }
  get chunks() {
    return this._collected;
  }
  _dequeue() {
    const chunk = this._queue[this._head];
    this._queue[this._head] = void 0;
    this._head = (this._head + 1) % this._capacity;
    this._count--;
    if (this._producerResolve) {
      const resolve = this._producerResolve;
      this._producerResolve = null;
      resolve();
    }
    return chunk;
  }
  [Symbol.asyncIterator]() {
    if (!this._streaming) {
      let index = 0;
      const collected = this._collected;
      return {
        async next() {
          if (index < collected.length) {
            return { done: false, value: collected[index++] };
          }
          return { done: true, value: void 0 };
        }
      };
    }
    const sink = this;
    return {
      async next() {
        while (sink._count === 0) {
          if (sink._error) throw sink._error;
          if (sink._done) return { done: true, value: void 0 };
          await new Promise((resolve) => {
            sink._consumerResolve = resolve;
          });
        }
        if (sink._error) throw sink._error;
        return { done: false, value: sink._dequeue() };
      }
    };
  }
  async collect() {
    const chunks = [];
    for await (const chunk of this) {
      chunks.push(chunk);
    }
    return chunks;
  }
};

// src/execution/pipeline.js
init_buffer_shim();
var PipelineGraph = class {
  constructor() {
    this.pipelines = /* @__PURE__ */ new Map();
    this.nextId = 1;
  }
  createPipeline(sink) {
    const id = this.nextId++;
    this.pipelines.set(id, {
      id,
      sink,
      source: null,
      dependencies: /* @__PURE__ */ new Set(),
      dependents: /* @__PURE__ */ new Set(),
      state: "PENDING",
      cancelled: false
    });
    return id;
  }
  addDependency(pipelineId, dependsOnId) {
    const pipeline = this.pipelines.get(pipelineId);
    const dependency = this.pipelines.get(dependsOnId);
    pipeline.dependencies.add(dependsOnId);
    dependency.dependents.add(pipelineId);
  }
  setSource(pipelineId, sourceGenerator) {
    const pipeline = this.pipelines.get(pipelineId);
    pipeline.source = sourceGenerator;
  }
  getReadyPipelines() {
    const ready = [];
    for (const [id, pipeline] of this.pipelines.entries()) {
      if (pipeline.state === "PENDING" && pipeline.dependencies.size === 0) {
        ready.push(pipeline);
      }
    }
    return ready;
  }
  markPipelineDone(pipelineId) {
    const pipeline = this.pipelines.get(pipelineId);
    pipeline.state = "DONE";
    for (const depId of pipeline.dependents) {
      const dependent = this.pipelines.get(depId);
      dependent.dependencies.delete(pipelineId);
    }
  }
  cancelPipeline(pipelineId) {
    const pipeline = this.pipelines.get(pipelineId);
    if (!pipeline || pipeline.cancelled) return;
    pipeline.cancelled = true;
  }
  isCancelled(pipelineId) {
    const pipeline = this.pipelines.get(pipelineId);
    return pipeline ? pipeline.cancelled : false;
  }
};
var CancelToken = class {
  constructor() {
    this.cancelled = false;
  }
  cancel() {
    this.cancelled = true;
  }
  get isCancelled() {
    return this.cancelled;
  }
};

// src/execution/scheduler.js
init_buffer_shim();
var TaskScheduler = class {
  constructor(concurrency = 4) {
    this.concurrency = concurrency;
  }
  async schedule(pipelineGraph) {
    let hasMoreWork = true;
    while (hasMoreWork) {
      const readyPipelines = pipelineGraph.getReadyPipelines();
      if (readyPipelines.length === 0) {
        let pendingCount = 0;
        for (const p of pipelineGraph.pipelines.values()) {
          if (p.state === "PENDING") pendingCount++;
        }
        if (pendingCount > 0) {
          throw new Error("Pipeline deadlock detected: pending pipelines with unresolved dependencies.");
        }
        break;
      }
      for (const p of readyPipelines) {
        p.state = "RUNNING";
      }
      await this.executePipelines(readyPipelines, pipelineGraph);
      for (const p of readyPipelines) {
        pipelineGraph.markPipelineDone(p.id);
      }
    }
  }
  async executePipelines(pipelines, graph) {
    const tasks = [];
    for (const p of pipelines) {
      if (p.source) {
        tasks.push(this.runPipelineSource(p, graph));
      }
    }
    await Promise.all(tasks);
  }
  async runPipelineSource(pipeline, graph) {
    const generator = pipeline.source();
    for await (const _ of generator) {
      if (pipeline.cancelled) break;
    }
  }
};

// src/execution/query-executor.js
init_config();

// src/storage/backend/memory-storage-backend.js
init_buffer_shim();

// src/storage/temp-space/memory-temp-space.js
init_buffer_shim();

// src/storage/storage-constants.js
init_buffer_shim();
var MEMORY_ROOT_PREFIX = "mem://query_engine";

// src/storage/temp-space/memory-temp-space.js
var MemoryTempSpace = class {
  constructor(options = {}) {
    this.rootDir = options.baseDir || MEMORY_ROOT_PREFIX;
    this.counters = /* @__PURE__ */ new Map();
  }
  allocate(category, label) {
    const seq = this.counters.get(category) || 0;
    this.counters.set(category, seq + 1);
    return `${this.rootDir}/${category}/${label}_${seq}`;
  }
  getRoot() {
    return this.rootDir;
  }
  cleanup() {
    this.counters.clear();
  }
};

// src/storage/page-store/memory-page-store.js
init_buffer_shim();
var MemoryPageStore = class {
  constructor() {
    this.pages = /* @__PURE__ */ new Map();
  }
  async write(pageId, chunk) {
    this.pages.set(pageId, chunk);
  }
  async read(pageId) {
    return this.pages.get(pageId) ?? null;
  }
  clear() {
    this.pages.clear();
  }
};

// src/storage/spill-manager/spill-manager.js
init_buffer_shim();

// src/storage/serializer.js
init_buffer_shim();
init_column();
init_dictionary_column();
init_chunk();
init_data_type();
init_sab_arena();
init_bitmap();
var COLUMN_KIND_FLAT = 0;
var COLUMN_KIND_DICTIONARY = 1;
var DATA_TYPE_TO_ID = {
  [DataType.BOOLEAN]: 0,
  [DataType.INT32]: 1,
  [DataType.INT64]: 2,
  [DataType.FLOAT64]: 3,
  [DataType.DECIMAL]: 4,
  [DataType.VARCHAR]: 5,
  [DataType.DATE]: 6,
  [DataType.TIMESTAMP]: 7
};
var ID_TO_DATA_TYPE = Object.fromEntries(
  Object.entries(DATA_TYPE_TO_ID).map(([k, v]) => [v, k])
);
var ChunkSerializer = class {
  static serialize(chunk) {
    const size = computeSize(chunk);
    const buf = import_buffer.Buffer.allocUnsafe(size);
    let offset = 0;
    buf.writeUInt32LE(chunk.size, offset);
    offset += 4;
    buf.writeUInt16LE(chunk.columns.length, offset);
    offset += 2;
    for (const col2 of chunk.columns) {
      const isDictionary = col2 instanceof DictionaryColumn;
      buf.writeUInt8(isDictionary ? COLUMN_KIND_DICTIONARY : COLUMN_KIND_FLAT, offset);
      offset += 1;
      buf.writeUInt8(DATA_TYPE_TO_ID[col2.dataType], offset);
      offset += 1;
      buf.writeUInt32LE(col2.length, offset);
      offset += 4;
      buf.writeUInt8(col2.hasNulls ? 1 : 0, offset);
      offset += 1;
      if (col2.hasNulls) {
        const bitmapWordCount2 = Math.ceil(col2.length / 32);
        buf.writeUInt32LE(bitmapWordCount2, offset);
        offset += 4;
        const bitmapBytes = bitmapWordCount2 * 4;
        import_buffer.Buffer.from(col2.nullBitmap.buffer, col2.nullBitmap.byteOffset, bitmapBytes).copy(buf, offset);
        offset += bitmapBytes;
      }
      if (isDictionary) {
        const indicesBytes = col2.length * 2;
        import_buffer.Buffer.from(col2.indices.buffer, col2.indices.byteOffset, indicesBytes).copy(buf, offset);
        offset += indicesBytes;
        const dictSize = col2.reverseDict.length;
        buf.writeUInt32LE(dictSize, offset);
        offset += 4;
        for (let i = 0; i < dictSize; i++) {
          const encoded = import_buffer.Buffer.from(col2.reverseDict[i], "utf8");
          buf.writeUInt32LE(encoded.length, offset);
          offset += 4;
          encoded.copy(buf, offset);
          offset += encoded.length;
        }
      } else if (isFixedWidth(col2.dataType)) {
        const bw = byteWidthFor(col2.dataType);
        const dataBytes = col2.length * bw;
        import_buffer.Buffer.from(col2.data.buffer, col2.data.byteOffset, dataBytes).copy(buf, offset);
        offset += dataBytes;
      } else {
        buf.writeUInt32LE(col2.stringBytesUsed, offset);
        offset += 4;
        const offsetsBytes = (col2.length + 1) * 4;
        import_buffer.Buffer.from(col2.offsets.buffer, col2.offsets.byteOffset, offsetsBytes).copy(buf, offset);
        offset += offsetsBytes;
        import_buffer.Buffer.from(col2.stringBytes.buffer, col2.stringBytes.byteOffset, col2.stringBytesUsed).copy(buf, offset);
        offset += col2.stringBytesUsed;
      }
    }
    return buf;
  }
  static deserialize(buffer, allocator = heapAllocator) {
    let offset = 0;
    const size = buffer.readUInt32LE(offset);
    offset += 4;
    const columnCount = buffer.readUInt16LE(offset);
    offset += 2;
    const columns = [];
    for (let c = 0; c < columnCount; c++) {
      const columnKind = buffer.readUInt8(offset);
      offset += 1;
      const dataTypeId = buffer.readUInt8(offset);
      offset += 1;
      const length = buffer.readUInt32LE(offset);
      offset += 4;
      const hasNulls = buffer.readUInt8(offset) !== 0;
      offset += 1;
      const dataType = ID_TO_DATA_TYPE[dataTypeId];
      let nullBitmap;
      if (hasNulls) {
        const storedWords = buffer.readUInt32LE(offset);
        offset += 4;
        nullBitmap = allocator.acquire(Uint32Array, storedWords);
        offset = copyBytesInto(buffer, offset, nullBitmap, storedWords * 4);
      } else {
        nullBitmap = allocator.acquire(Uint32Array, bitmapWordCount(Math.max(length, 1)));
      }
      if (columnKind === COLUMN_KIND_DICTIONARY) {
        const indices = allocator.acquire(Uint16Array, length);
        offset = copyBytesInto(buffer, offset, indices, length * 2);
        const dictSize = buffer.readUInt32LE(offset);
        offset += 4;
        const reverseDict = [];
        for (let d = 0; d < dictSize; d++) {
          const strLen = buffer.readUInt32LE(offset);
          offset += 4;
          reverseDict.push(buffer.toString("utf8", offset, offset + strLen));
          offset += strLen;
        }
        columns.push(DictionaryColumn.fromParts({ indices, reverseDict, nullBitmap, length, hasNulls, allocator }));
      } else if (isFixedWidth(dataType)) {
        const data = allocator.acquire(typedArrayCtorFor(dataType), length);
        offset = copyBytesInto(buffer, offset, data, length * byteWidthFor(dataType));
        columns.push(Column.fromParts({ dataType, data, nullBitmap, length, hasNulls, allocator }));
      } else {
        const stringBytesUsed = buffer.readUInt32LE(offset);
        offset += 4;
        const offsets = allocator.acquire(Uint32Array, length + 1);
        offset = copyBytesInto(buffer, offset, offsets, (length + 1) * 4);
        const stringBytes = allocator.acquire(Uint8Array, stringBytesUsed);
        offset = copyBytesInto(buffer, offset, stringBytes, stringBytesUsed);
        columns.push(Column.fromParts({ dataType, offsets, stringBytes, stringBytesUsed, nullBitmap, length, hasNulls, allocator }));
      }
    }
    return new DataChunk(columns, size);
  }
};
function copyBytesInto(buffer, offset, view, byteLength) {
  const dest = new Uint8Array(view.buffer, view.byteOffset, byteLength);
  dest.set(buffer.subarray(offset, offset + byteLength));
  return offset + byteLength;
}
function computeSize(chunk) {
  let total = 4 + 2;
  for (const col2 of chunk.columns) {
    const isDictionary = col2 instanceof DictionaryColumn;
    total += 1 + 1 + 4 + 1;
    if (col2.hasNulls) {
      const bitmapWordCount2 = Math.ceil(col2.length / 32);
      total += 4 + bitmapWordCount2 * 4;
    }
    if (isDictionary) {
      total += col2.length * 2;
      total += 4;
      for (let i = 0; i < col2.reverseDict.length; i++) {
        total += 4 + import_buffer.Buffer.byteLength(col2.reverseDict[i], "utf8");
      }
    } else if (isFixedWidth(col2.dataType)) {
      total += col2.length * byteWidthFor(col2.dataType);
    } else {
      total += 4 + (col2.length + 1) * 4 + col2.stringBytesUsed;
    }
  }
  return total;
}

// src/storage/spill-manager/spill-manager.js
var LENGTH_HEADER_BYTES = 4;
var SpillManager = class {
  constructor(storage) {
    this.storage = storage;
  }
  async appendChunk(partitionId, chunk) {
    if (!chunk || chunk.size === 0) return;
    const data = ChunkSerializer.serialize(chunk);
    const header = import_buffer.Buffer.allocUnsafe(LENGTH_HEADER_BYTES);
    header.writeUInt32LE(data.length, 0);
    await this.storage.append(partitionId, import_buffer.Buffer.concat([header, data]));
  }
  async *readChunks(partitionId) {
    const fileBuffer = await this.storage.read(partitionId);
    if (!fileBuffer) return;
    let offset = 0;
    while (offset < fileBuffer.length) {
      const chunkLength = fileBuffer.readUInt32LE(offset);
      offset += LENGTH_HEADER_BYTES;
      const chunkData = fileBuffer.subarray(offset, offset + chunkLength);
      yield ChunkSerializer.deserialize(chunkData);
      offset += chunkLength;
    }
  }
  async clearPartition(partitionId) {
    await this.storage.remove(partitionId);
  }
  async clearAll() {
    await this.storage.removeAll();
  }
  hasSpilled(partitionId) {
    return this.storage.exists(partitionId);
  }
};

// src/storage/spill-manager/memory-storage.js
init_buffer_shim();
var MemoryStorage = class {
  constructor() {
    this.store = /* @__PURE__ */ new Map();
  }
  async append(partitionId, buffer) {
    if (!this.store.has(partitionId)) this.store.set(partitionId, []);
    this.store.get(partitionId).push(import_buffer.Buffer.from(buffer));
  }
  async read(partitionId) {
    const buffers = this.store.get(partitionId);
    if (!buffers || buffers.length === 0) return null;
    return import_buffer.Buffer.concat(buffers);
  }
  exists(partitionId) {
    const buffers = this.store.get(partitionId);
    return !!buffers && buffers.length > 0;
  }
  async remove(partitionId) {
    this.store.delete(partitionId);
  }
  async removeAll() {
    this.store.clear();
  }
};

// src/storage/backend/memory-storage-backend.js
var MemoryStorageBackend = class {
  constructor(options = {}) {
    this.options = options;
  }
  createTempSpace() {
    return new MemoryTempSpace(this.options);
  }
  createPageStore() {
    return new MemoryPageStore();
  }
  createSpillManager() {
    return new SpillManager(new MemoryStorage());
  }
};

// src/execution/builders/source-builders.js
init_buffer_shim();

// src/execution/operators/scan.js
init_buffer_shim();
var ScanOperator = class {
  constructor(table, projectedColumns) {
    this.table = table;
    this.projectedColumns = projectedColumns || null;
  }
  async init() {
  }
  async *scan() {
    for await (const chunk of this.table.scan()) {
      if (this.projectedColumns) {
        yield chunk.project(this.projectedColumns);
      } else {
        yield chunk;
      }
    }
  }
  estimatedRows() {
    return this.table.rowCount();
  }
};

// src/execution/operators/index-scan.js
init_buffer_shim();
init_column();
init_chunk();
init_config();
var IndexScanOperator = class {
  constructor(btreeIndex, table, scanType, scanKey, scanLow, scanHigh, lowInc, highInc, projectedColumns) {
    this.btreeIndex = btreeIndex;
    this.table = table;
    this.scanType = scanType;
    this.scanKey = scanKey;
    this.scanLow = scanLow;
    this.scanHigh = scanHigh;
    this.lowInc = lowInc;
    this.highInc = highInc;
    this.projectedColumns = projectedColumns;
  }
  async *scan() {
    let locations;
    if (this.scanType === "point") {
      locations = this.btreeIndex.search(this.scanKey);
    } else {
      locations = [...this.btreeIndex.range(this.scanLow, this.scanHigh, this.lowInc, this.highInc)];
    }
    if (locations.length === 0) return;
    const pageGroups = /* @__PURE__ */ new Map();
    for (const loc of locations) {
      let group = pageGroups.get(loc.pageId);
      if (!group) {
        group = [];
        pageGroups.set(loc.pageId, group);
      }
      group.push(loc.rowIndex);
    }
    const schema = this.table.getSchema();
    const outputSchema = this.projectedColumns ? this.projectedColumns.map((i) => schema[i]) : schema;
    let pendingRows = [];
    for (const [pageId, rowIndices] of pageGroups) {
      const page = await this.table.bufferPool.fetchPage(pageId, true);
      for (const rowIdx of rowIndices) {
        const row = [];
        if (this.projectedColumns) {
          for (const colIdx of this.projectedColumns) {
            row.push(page.columns[colIdx].get(rowIdx));
          }
        } else {
          for (let c = 0; c < page.columns.length; c++) {
            row.push(page.columns[c].get(rowIdx));
          }
        }
        pendingRows.push(row);
        if (pendingRows.length >= Config.flushBatchSize) {
          yield this._buildChunk(pendingRows, outputSchema);
          pendingRows = [];
        }
      }
    }
    if (pendingRows.length > 0) {
      yield this._buildChunk(pendingRows, outputSchema);
    }
  }
  _buildChunk(rows, schema) {
    const colCount = schema.length;
    const columns = new Array(colCount);
    for (let c = 0; c < colCount; c++) {
      const col2 = new Column(schema[c].dataType, rows.length);
      for (let r = 0; r < rows.length; r++) {
        col2.set(r, rows[r][c]);
      }
      col2.length = rows.length;
      columns[c] = col2;
    }
    return new DataChunk(columns, rows.length);
  }
};

// src/execution/builders/source-builders.js
init_chunk();
async function buildScan(executor, node) {
  const storage = executor.catalog.getTableStorage(node.table);
  if (!storage) throw new Error(`No storage for table: ${node.table}`);
  const schema = storage.getSchema();
  const projectedColumns = executor.resolveProjectedColumnIndexes(schema, node.columns);
  const outputSchema = projectedColumns ? projectedColumns.map((i) => schema[i]) : schema;
  const finalSchema = outputSchema.map((c) => ({ ...c, tableAlias: node.alias || node.table }));
  const columnMapping = executor.buildSchemaMapping(finalSchema, node.alias || node.table);
  return {
    schema: finalSchema,
    columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const scanOp = new ScanOperator(storage, projectedColumns);
      graph.setSource(currentPipelineId, async function* () {
        for await (const chunk of scanOp.scan()) {
          if (currentSink.cancelToken?.isCancelled) break;
          await currentSink.consume(chunk);
          yield chunk;
        }
        if (currentSink.finalize) await currentSink.finalize();
      });
    }
  };
}
async function buildIndexScan(executor, node) {
  const storage = executor.catalog.getTableStorage(node.table);
  if (!storage) throw new Error(`No storage for table: ${node.table}`);
  const btree = executor.catalog.getIndexForColumn(node.table, node.columnName);
  if (!btree) throw new Error(`No index for ${node.table}.${node.columnName}`);
  const schema = storage.getSchema();
  const projectedColumns = executor.resolveProjectedColumnIndexes(schema, node.columns);
  const outputSchema = projectedColumns ? projectedColumns.map((i) => schema[i]) : schema;
  const finalSchema = outputSchema.map((c) => ({ ...c, tableAlias: node.alias || node.table }));
  const columnMapping = executor.buildSchemaMapping(finalSchema, node.alias || node.table);
  return {
    schema: finalSchema,
    columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const scanOp = new IndexScanOperator(
        btree,
        storage,
        node.scanType,
        node.scanKey,
        node.scanLow,
        node.scanHigh,
        node.lowInc,
        node.highInc,
        projectedColumns
      );
      graph.setSource(currentPipelineId, async function* () {
        for await (const chunk of scanOp.scan()) {
          if (currentSink.cancelToken?.isCancelled) break;
          await currentSink.consume(chunk);
          yield chunk;
        }
        if (currentSink.finalize) await currentSink.finalize();
      });
    }
  };
}
async function buildSingleRow(executor, node) {
  return {
    schema: [],
    columnMapping: /* @__PURE__ */ new Map(),
    register: (graph, currentPipelineId, currentSink) => {
      graph.setSource(currentPipelineId, async function* () {
        const chunk = new DataChunk([], 1);
        await currentSink.consume(chunk);
        yield chunk;
        if (currentSink.finalize) await currentSink.finalize();
      });
    }
  };
}
async function buildEmpty(executor, node) {
  const child = await executor.buildPipeline(node.children[0]);
  return {
    schema: child.schema,
    columnMapping: child.columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      graph.setSource(currentPipelineId, async function* () {
      });
    }
  };
}

// src/execution/builders/pipeline-builders.js
init_buffer_shim();

// src/execution/operators/sort.js
init_buffer_shim();
init_column();
init_chunk();

// src/utils/priority-queue.js
init_buffer_shim();
var PriorityQueue = class {
  constructor(comparator = (a, b) => a - b) {
    this._heap = [];
    this._comparator = comparator;
  }
  get size() {
    return this._heap.length;
  }
  isEmpty() {
    return this.size === 0;
  }
  peek() {
    return this._heap[0];
  }
  push(value) {
    this._heap.push(value);
    this._siftUp();
  }
  pop() {
    if (this.isEmpty()) return null;
    const poppedValue = this.peek();
    const bottom = this.size - 1;
    if (bottom > 0) {
      this._heap[0] = this._heap[bottom];
    }
    this._heap.pop();
    this._siftDown();
    return poppedValue;
  }
  _parent(i) {
    return (i + 1 >>> 1) - 1;
  }
  _left(i) {
    return (i << 1) + 1;
  }
  _right(i) {
    return i + 1 << 1;
  }
  _siftUp() {
    let node = this.size - 1;
    while (node > 0 && this._compare(node, this._parent(node))) {
      this._swap(node, this._parent(node));
      node = this._parent(node);
    }
  }
  _siftDown() {
    let node = 0;
    while (this._left(node) < this.size && this._compare(this._left(node), node) || this._right(node) < this.size && this._compare(this._right(node), node)) {
      let maxChild = this._right(node) < this.size && this._compare(this._right(node), this._left(node)) ? this._right(node) : this._left(node);
      this._swap(node, maxChild);
      node = maxChild;
    }
  }
  _compare(i, j) {
    return this._comparator(this._heap[i], this._heap[j]) < 0;
  }
  _swap(i, j) {
    const temp = this._heap[i];
    this._heap[i] = this._heap[j];
    this._heap[j] = temp;
  }
};

// src/execution/operators/sort.js
init_config();
var SortOperator = class {
  constructor(keyExtractors, limit, offset, spillManager) {
    this.keyExtractors = keyExtractors;
    this.limit = limit ?? null;
    this.offset = offset || 0;
    this.topN = this.limit !== null ? this.limit + this.offset : null;
    this.rows = [];
    this.schema = null;
    this.spillManager = spillManager;
    this.runCount = 0;
  }
  async init() {
  }
  async consume(chunk) {
    if (!this.schema) {
      this.schema = chunk.columns.map((c) => c.dataType);
    }
    const chunkRows = new Array(chunk.size);
    for (let i = 0; i < chunk.size; i++) {
      const rowIdx = chunk.activeRowIndex(i);
      const row = new Array(chunk.columns.length);
      for (let c = 0; c < chunk.columns.length; c++) {
        row[c] = chunk.columns[c].get(rowIdx);
      }
      const sortKeys = new Array(this.keyExtractors.length);
      for (let k = 0; k < this.keyExtractors.length; k++) {
        sortKeys[k] = this.keyExtractors[k].eval(chunk, rowIdx);
      }
      chunkRows[i] = { row, sortKeys };
    }
    this.rows.push(...chunkRows);
    if (this.topN && this.runCount === 0 && this.rows.length > this.topN * 4) {
      this.rows.sort((a, b) => this.compareRows(a, b));
      this.rows.length = this.topN;
    }
    if (this.rows.length >= Config.memoryLimit) {
      await this.spillCurrentRun();
    }
  }
  async spillCurrentRun() {
    if (this.rows.length === 0) return;
    this.rows.sort((a, b) => this.compareRows(a, b));
    if (this.topN) {
      this.rows.length = Math.min(this.rows.length, this.topN);
    }
    const chunk = this.rowsToChunk(this.rows);
    await this.spillManager.appendChunk(`run_${this.runCount}`, chunk);
    this.runCount++;
    this.rows = [];
  }
  async finalize() {
    if (this.runCount === 0) {
      this.rows.sort((a, b) => this.compareRows(a, b));
      if (this.topN) {
        this.rows.length = Math.min(this.rows.length, this.topN);
      }
      if (this.offset > 0) {
        this.rows = this.rows.slice(this.offset);
      }
      if (this.rows.length === 0) return [];
      const chunk = this.rowsToChunk(this.rows);
      await this.spillManager.clearAll();
      return [chunk];
    }
    if (this.rows.length > 0) {
      await this.spillCurrentRun();
    }
    const iterators = [];
    for (let i = 0; i < this.runCount; i++) {
      iterators.push(this.spillManager.readChunks(`run_${i}`));
    }
    const pq = new PriorityQueue((a, b) => this.compareRows(a.item, b.item));
    const states = new Array(this.runCount);
    for (let i = 0; i < this.runCount; i++) {
      const iter = iterators[i];
      const next = await iter.next();
      if (!next.done && next.value.size > 0) {
        states[i] = {
          iter,
          chunk: next.value,
          index: 0,
          chunkItems: this.chunkToItems(next.value)
        };
        pq.push({ item: states[i].chunkItems[0], runIndex: i });
        states[i].index = 1;
      }
    }
    const resultChunks = [];
    let outRows = [];
    let count2 = 0;
    let skipped = 0;
    while (!pq.isEmpty()) {
      if (this.topN && count2 >= this.topN) break;
      const { item, runIndex } = pq.pop();
      count2++;
      if (skipped < this.offset) {
        skipped++;
      } else {
        outRows.push(item);
      }
      if (outRows.length >= Config.flushBatchSize) {
        resultChunks.push(this.rowsToChunk(outRows));
        outRows = [];
      }
      const state = states[runIndex];
      if (state.index < state.chunkItems.length) {
        pq.push({ item: state.chunkItems[state.index], runIndex });
        state.index++;
      } else {
        const next = await state.iter.next();
        if (!next.done && next.value.size > 0) {
          state.chunk = next.value;
          state.index = 0;
          state.chunkItems = this.chunkToItems(state.chunk);
          pq.push({ item: state.chunkItems[state.index], runIndex });
          state.index++;
        }
      }
    }
    if (outRows.length > 0) {
      resultChunks.push(this.rowsToChunk(outRows));
    }
    await this.spillManager.clearAll();
    return resultChunks;
  }
  chunkToItems(chunk) {
    const items = new Array(chunk.size);
    for (let i = 0; i < chunk.size; i++) {
      const rowIdx = chunk.activeRowIndex(i);
      const row = new Array(chunk.columns.length);
      for (let c = 0; c < chunk.columns.length; c++) {
        row[c] = chunk.columns[c].get(rowIdx);
      }
      const sortKeys = new Array(this.keyExtractors.length);
      for (let k = 0; k < this.keyExtractors.length; k++) {
        sortKeys[k] = this.keyExtractors[k].eval(chunk, rowIdx);
      }
      items[i] = { row, sortKeys };
    }
    return items;
  }
  rowsToChunk(items) {
    if (items.length === 0) return new DataChunk([], 0);
    const colCount = items[0].row.length;
    const columns = new Array(colCount);
    for (let c = 0; c < colCount; c++) {
      const col2 = new Column(this.schema?.[c] || "VARCHAR", items.length);
      for (let r = 0; r < items.length; r++) {
        col2.set(r, items[r].row[c]);
      }
      col2.length = items.length;
      columns[c] = col2;
    }
    return new DataChunk(columns, items.length);
  }
  compareRows(a, b) {
    for (let i = 0; i < this.keyExtractors.length; i++) {
      let v1 = a.sortKeys[i];
      let v2 = b.sortKeys[i];
      if (typeof v1 === "bigint") v1 = Number(v1);
      if (typeof v2 === "bigint") v2 = Number(v2);
      if (v1 === null && v2 !== null) return 1;
      if (v1 !== null && v2 === null) return -1;
      if (v1 === null && v2 === null) continue;
      if (v1 < v2) return this.keyExtractors[i].direction === "ASC" ? -1 : 1;
      if (v1 > v2) return this.keyExtractors[i].direction === "ASC" ? 1 : -1;
    }
    return 0;
  }
};
var LimitOperator = class {
  constructor(limit, offset = 0) {
    this.limit = limit;
    this.offset = offset;
    this.seen = 0;
    this.emitted = 0;
    this.chunks = [];
    this.schema = null;
    this.done = false;
  }
  async init() {
  }
  async consume(chunk) {
    if (this.done) return;
    if (!this.schema) {
      this.schema = chunk.columns.map((c) => c.dataType);
    }
    const chunkStart = this.seen;
    const chunkEnd = this.seen + chunk.size;
    this.seen = chunkEnd;
    if (chunkEnd <= this.offset) return;
    const startInChunk = Math.max(0, this.offset - chunkStart);
    const remaining = this.limit - this.emitted;
    if (remaining <= 0) {
      this.done = true;
      return;
    }
    const endInChunk = Math.min(chunk.size, startInChunk + remaining);
    const count2 = endInChunk - startInChunk;
    if (count2 <= 0) return;
    if (startInChunk === 0 && count2 === chunk.size && !chunk.selectionVector) {
      this.chunks.push(chunk);
    } else {
      const sv = new Uint32Array(count2);
      for (let i = 0; i < count2; i++) {
        sv[i] = chunk.activeRowIndex(startInChunk + i);
      }
      const result = new DataChunk(chunk.columns, count2);
      result.setSelectionVector(sv, count2);
      this.chunks.push(result);
    }
    this.emitted += count2;
    if (this.emitted >= this.limit) {
      this.done = true;
    }
  }
  async finalize() {
    if (this.chunks.length === 0) return [];
    return this.chunks.map((c) => c.selectionVector ? c.flatten() : c);
  }
};

// src/execution/operators/distinct.js
init_buffer_shim();
init_column();
init_chunk();
var DistinctOperator = class {
  constructor() {
    this.seen = /* @__PURE__ */ new Set();
    this.schema = null;
  }
  async init() {
  }
  async process(chunk) {
    if (!this.schema) {
      this.schema = chunk.columns.map((c) => c.dataType);
    }
    const sv = new Uint32Array(chunk.size);
    let count2 = 0;
    for (let i = 0; i < chunk.size; i++) {
      const rowIdx = chunk.activeRowIndex(i);
      let key = "";
      for (let c = 0; c < chunk.columns.length; c++) {
        if (c > 0) key += "|";
        key += String(chunk.columns[c].get(rowIdx));
      }
      if (!this.seen.has(key)) {
        this.seen.add(key);
        sv[count2++] = rowIdx;
      }
    }
    if (count2 === 0) return new DataChunk(chunk.columns, 0);
    if (count2 === chunk.size) return chunk;
    const result = new DataChunk(chunk.columns, count2);
    result.setSelectionVector(sv.slice(0, count2), count2);
    return result;
  }
  async consume(chunk) {
    if (!this._legacyChunks) this._legacyChunks = [];
    const result = await this.process(chunk);
    if (result.size > 0) {
      this._legacyChunks.push(result.selectionVector ? result.flatten() : result);
    }
  }
  async finalize() {
    return this._legacyChunks || [];
  }
};

// src/execution/operators/union.js
init_buffer_shim();
init_column();
init_chunk();
var UnionOperator = class {
  constructor(isAll) {
    this.isAll = isAll;
    this.seen = isAll ? null : /* @__PURE__ */ new Set();
    this.schema = null;
  }
  async init() {
  }
  async process(chunk) {
    if (this.isAll) return chunk;
    if (!this.schema) {
      this.schema = chunk.columns.map((c) => c.dataType);
    }
    const sv = new Uint32Array(chunk.size);
    let count2 = 0;
    for (let i = 0; i < chunk.size; i++) {
      const rowIdx = chunk.activeRowIndex(i);
      let key = "";
      for (let c = 0; c < chunk.columns.length; c++) {
        if (c > 0) key += "|";
        key += String(chunk.columns[c].get(rowIdx));
      }
      if (!this.seen.has(key)) {
        this.seen.add(key);
        sv[count2++] = rowIdx;
      }
    }
    if (count2 === 0) return new DataChunk(chunk.columns, 0);
    if (count2 === chunk.size) return chunk;
    const result = new DataChunk(chunk.columns, count2);
    result.setSelectionVector(sv.slice(0, count2), count2);
    return result;
  }
  async consume(chunk) {
    if (!this._legacyChunks) this._legacyChunks = [];
    const result = await this.process(chunk);
    if (result.size > 0) {
      this._legacyChunks.push(result.selectionVector ? result.flatten() : result);
    }
  }
  async finalize() {
    return this._legacyChunks || [];
  }
};

// src/execution/operators/window.js
init_buffer_shim();
init_chunk();
init_column();
var WindowOperator = class {
  constructor(windowExprs, childSchema, childColumnMapping, compileExpressionFn) {
    this.windowExprs = windowExprs;
    this.childSchema = childSchema;
    this.childColumnMapping = childColumnMapping;
    this.compileExpression = compileExpressionFn;
  }
  async execute(chunks) {
    const allRows = [];
    for (const chunk of chunks) {
      for (let r = 0; r < chunk.size; r++) {
        const row = [];
        for (let c = 0; c < chunk.columns.length; c++) {
          row.push(chunk.columns[c].get(chunk.activeRowIndex ? chunk.activeRowIndex(r) : r));
        }
        allRows.push(row);
      }
    }
    if (allRows.length === 0) return [];
    const windowResults = [];
    for (const wExpr of this.windowExprs) {
      windowResults.push(this.computeWindow(wExpr, allRows, chunks));
    }
    const colCount = this.childSchema.length + this.windowExprs.length;
    const resultCol = [];
    for (let c = 0; c < this.childSchema.length; c++) {
      const col2 = new Column(this.childSchema[c].dataType, allRows.length);
      for (let r = 0; r < allRows.length; r++) {
        col2.set(r, allRows[r][c]);
      }
      col2.length = allRows.length;
      resultCol.push(col2);
    }
    for (let w = 0; w < windowResults.length; w++) {
      const dt = this.windowExprs[w].resultType || "INT64";
      const col2 = new Column(dt === "INT64" ? "FLOAT64" : dt, allRows.length);
      for (let r = 0; r < allRows.length; r++) {
        col2.set(r, windowResults[w][r]);
      }
      col2.length = allRows.length;
      resultCol.push(col2);
    }
    return [new DataChunk(resultCol, allRows.length)];
  }
  computeWindow(wExpr, allRows, chunks) {
    const partitionBy = wExpr.partitionBy.map((e) => this.compileExpression(e, this.childColumnMapping));
    const orderKeys = wExpr.orderBy.map((ok) => ({
      eval: this.compileExpression(ok.expr, this.childColumnMapping),
      direction: ok.direction || "ASC"
    }));
    const tempChunk = chunks.length > 0 ? chunks[0] : null;
    const getVal = (rowIdx, evalFn) => {
      let offset = 0;
      for (const chunk of chunks) {
        if (rowIdx < offset + chunk.size) {
          return evalFn(chunk, rowIdx - offset);
        }
        offset += chunk.size;
      }
      return null;
    };
    const partitions = this.partitionRows(allRows, partitionBy, chunks);
    if (orderKeys.length > 0) {
      for (const partition of partitions) {
        partition.sort((a, b) => {
          for (const key of orderKeys) {
            const va = getVal(a, key.eval);
            const vb = getVal(b, key.eval);
            const an = va === null || va === void 0;
            const bn = vb === null || vb === void 0;
            if (an && bn) continue;
            if (an) return 1;
            if (bn) return -1;
            const cmp = this.compareValues(va, vb);
            if (cmp !== 0) return key.direction === "DESC" ? -cmp : cmp;
          }
          return 0;
        });
      }
    }
    const result = new Array(allRows.length);
    const name = wExpr.name.toUpperCase();
    for (const partition of partitions) {
      switch (name) {
        case "ROW_NUMBER":
          for (let i = 0; i < partition.length; i++) {
            result[partition[i]] = i + 1;
          }
          break;
        case "RANK": {
          let rank = 1;
          for (let i = 0; i < partition.length; i++) {
            if (i > 0 && !this.sameOrderKey(partition[i], partition[i - 1], orderKeys, chunks)) {
              rank = i + 1;
            }
            result[partition[i]] = rank;
          }
          break;
        }
        case "DENSE_RANK": {
          let rank = 1;
          for (let i = 0; i < partition.length; i++) {
            if (i > 0 && !this.sameOrderKey(partition[i], partition[i - 1], orderKeys, chunks)) {
              rank++;
            }
            result[partition[i]] = rank;
          }
          break;
        }
        case "LAG": {
          const valueEval = wExpr.args.length > 0 ? this.compileExpression(wExpr.args[0], this.childColumnMapping) : null;
          const lagOffset = wExpr.args.length > 1 ? wExpr.args[1].value : 1;
          const defaultVal = wExpr.args.length > 2 ? wExpr.args[2].value : null;
          for (let i = 0; i < partition.length; i++) {
            const srcIdx = i - lagOffset;
            if (srcIdx >= 0 && srcIdx < partition.length) {
              result[partition[i]] = valueEval ? getVal(partition[srcIdx], valueEval) : null;
            } else {
              result[partition[i]] = defaultVal;
            }
          }
          break;
        }
        case "LEAD": {
          const valueEval = wExpr.args.length > 0 ? this.compileExpression(wExpr.args[0], this.childColumnMapping) : null;
          const leadOffset = wExpr.args.length > 1 ? wExpr.args[1].value : 1;
          const defaultVal = wExpr.args.length > 2 ? wExpr.args[2].value : null;
          for (let i = 0; i < partition.length; i++) {
            const srcIdx = i + leadOffset;
            if (srcIdx >= 0 && srcIdx < partition.length) {
              result[partition[i]] = valueEval ? getVal(partition[srcIdx], valueEval) : null;
            } else {
              result[partition[i]] = defaultVal;
            }
          }
          break;
        }
        case "SUM": {
          const valueEval = this.compileExpression(wExpr.args[0], this.childColumnMapping);
          if (orderKeys.length === 0) {
            let total = 0;
            let count2 = 0;
            for (let i = 0; i < partition.length; i++) {
              const v = getVal(partition[i], valueEval);
              if (v !== null && v !== void 0) {
                total += typeof v === "bigint" ? Number(v) : v;
                count2++;
              }
            }
            const sum2 = count2 === 0 ? null : total;
            for (let i = 0; i < partition.length; i++) result[partition[i]] = sum2;
          } else {
            let total = 0;
            let count2 = 0;
            for (let i = 0; i < partition.length; i++) {
              const v = getVal(partition[i], valueEval);
              if (v !== null && v !== void 0) {
                total += typeof v === "bigint" ? Number(v) : v;
                count2++;
              }
              result[partition[i]] = count2 === 0 ? null : total;
            }
          }
          break;
        }
        case "AVG": {
          const valueEval = this.compileExpression(wExpr.args[0], this.childColumnMapping);
          if (orderKeys.length === 0) {
            let total = 0;
            let count2 = 0;
            for (let i = 0; i < partition.length; i++) {
              const v = getVal(partition[i], valueEval);
              if (v !== null && v !== void 0) {
                total += typeof v === "bigint" ? Number(v) : v;
                count2++;
              }
            }
            const avg2 = count2 === 0 ? null : total / count2;
            for (let i = 0; i < partition.length; i++) result[partition[i]] = avg2;
          } else {
            let total = 0;
            let count2 = 0;
            for (let i = 0; i < partition.length; i++) {
              const v = getVal(partition[i], valueEval);
              if (v !== null && v !== void 0) {
                total += typeof v === "bigint" ? Number(v) : v;
                count2++;
              }
              result[partition[i]] = count2 === 0 ? null : total / count2;
            }
          }
          break;
        }
        case "COUNT":
        case "COUNT_STAR": {
          const valueEval = wExpr.args.length > 0 ? this.compileExpression(wExpr.args[0], this.childColumnMapping) : null;
          if (orderKeys.length === 0) {
            let total = 0;
            for (let i = 0; i < partition.length; i++) {
              if (valueEval) {
                const v = getVal(partition[i], valueEval);
                if (v !== null) total++;
              } else {
                total++;
              }
            }
            for (let i = 0; i < partition.length; i++) result[partition[i]] = total;
          } else {
            let count2 = 0;
            for (let i = 0; i < partition.length; i++) {
              if (valueEval) {
                const v = getVal(partition[i], valueEval);
                if (v !== null) count2++;
              } else {
                count2++;
              }
              result[partition[i]] = count2;
            }
          }
          break;
        }
        case "MIN": {
          const valueEval = this.compileExpression(wExpr.args[0], this.childColumnMapping);
          if (orderKeys.length === 0) {
            let min2 = null;
            for (let i = 0; i < partition.length; i++) {
              const v = getVal(partition[i], valueEval);
              if (v !== null && (min2 === null || v < min2)) min2 = v;
            }
            for (let i = 0; i < partition.length; i++) result[partition[i]] = min2;
          } else {
            let min2 = null;
            for (let i = 0; i < partition.length; i++) {
              const v = getVal(partition[i], valueEval);
              if (v !== null && (min2 === null || v < min2)) min2 = v;
              result[partition[i]] = min2;
            }
          }
          break;
        }
        case "MAX": {
          const valueEval = this.compileExpression(wExpr.args[0], this.childColumnMapping);
          if (orderKeys.length === 0) {
            let max2 = null;
            for (let i = 0; i < partition.length; i++) {
              const v = getVal(partition[i], valueEval);
              if (v !== null && (max2 === null || v > max2)) max2 = v;
            }
            for (let i = 0; i < partition.length; i++) result[partition[i]] = max2;
          } else {
            let max2 = null;
            for (let i = 0; i < partition.length; i++) {
              const v = getVal(partition[i], valueEval);
              if (v !== null && (max2 === null || v > max2)) max2 = v;
              result[partition[i]] = max2;
            }
          }
          break;
        }
        default:
          for (let i = 0; i < partition.length; i++) {
            result[partition[i]] = null;
          }
      }
    }
    return result;
  }
  partitionRows(allRows, partitionEvals, chunks) {
    if (partitionEvals.length === 0) {
      return [allRows.map((_, i) => i)];
    }
    const getVal = (rowIdx, evalFn) => {
      let offset = 0;
      for (const chunk of chunks) {
        if (rowIdx < offset + chunk.size) {
          return evalFn(chunk, rowIdx - offset);
        }
        offset += chunk.size;
      }
      return null;
    };
    const groups = /* @__PURE__ */ new Map();
    for (let i = 0; i < allRows.length; i++) {
      const key = partitionEvals.map((e) => {
        const v = getVal(i, e);
        return typeof v === "bigint" ? Number(v) : v;
      }).join("|");
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(i);
    }
    return [...groups.values()];
  }
  sameOrderKey(idxA, idxB, orderKeys, chunks) {
    const getVal = (rowIdx, evalFn) => {
      let offset = 0;
      for (const chunk of chunks) {
        if (rowIdx < offset + chunk.size) {
          return evalFn(chunk, rowIdx - offset);
        }
        offset += chunk.size;
      }
      return null;
    };
    for (const key of orderKeys) {
      const va = getVal(idxA, key.eval);
      const vb = getVal(idxB, key.eval);
      if (this.compareValues(va, vb) !== 0) return false;
    }
    return true;
  }
  compareValues(a, b) {
    const na = typeof a === "bigint" ? Number(a) : a;
    const nb = typeof b === "bigint" ? Number(b) : b;
    if (na === null && nb === null) return 0;
    if (na === null) return 1;
    if (nb === null) return -1;
    if (na < nb) return -1;
    if (na > nb) return 1;
    return 0;
  }
};

// src/execution/builders/pipeline-builders.js
init_expression_binder();

// src/execution/builders/builder-utils.js
init_buffer_shim();
function combinedMappingOf(...schemas) {
  const mapping = /* @__PURE__ */ new Map();
  let idx = 0;
  for (const schema of schemas) {
    for (const col2 of schema) {
      const key = `${col2.tableAlias}.${col2.name}`.toUpperCase();
      mapping.set(key, idx);
      if (!mapping.has(col2.name.toUpperCase())) {
        mapping.set(col2.name.toUpperCase(), idx);
      }
      idx++;
    }
  }
  return mapping;
}
function registerBufferedChild(graph, currentPipelineId, compiled) {
  const chunks = [];
  const sink = {
    consume: async (chunk) => {
      chunks.push(chunk);
    },
    finalize: async () => {
    }
  };
  const pipelineId = graph.createPipeline(sink);
  compiled.register(graph, pipelineId, sink);
  graph.addDependency(currentPipelineId, pipelineId);
  return chunks;
}

// src/execution/builders/pipeline-builders.js
async function buildFilter(executor, node) {
  const child = await executor.buildPipeline(node.children[0]);
  const evalFn = compileExpression(node.condition, child.columnMapping);
  const parallelDispatch = executor.parallelDispatch;
  return {
    schema: child.schema,
    columnMapping: child.columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const filterOp = new FilterOperator(node.condition, evalFn, child.columnMapping, parallelDispatch);
      const childSink = {
        get cancelToken() {
          return currentSink.cancelToken;
        },
        async consume(chunk) {
          if (this.cancelToken?.isCancelled) return;
          const filtered = await filterOp.process(chunk);
          if (filtered && filtered.size > 0) {
            await currentSink.consume(filtered);
          }
        },
        async finalize() {
          if (currentSink.finalize) await currentSink.finalize();
        }
      };
      child.register(graph, currentPipelineId, childSink);
    }
  };
}
async function buildProject(executor, node) {
  const child = await executor.buildPipeline(node.children[0]);
  const evaluators = node.expressions.map((expr2) => compileExpression(expr2, child.columnMapping));
  const resultTypes = node.expressions.map((expr2) => executor.normalizeExecType(expr2?.dataType || expr2?.resultType || "VARCHAR"));
  const schema = node.expressions.map((expr2, i) => ({
    name: expr2?.outputName || expr2?.alias || expr2?.name || expr2?.columnName || `col${i}`,
    dataType: executor.normalizeExecType(expr2?.dataType || expr2?.resultType || "VARCHAR"),
    tableAlias: ""
  }));
  const columnMapping = executor.buildSchemaMapping(schema, "");
  return {
    schema,
    columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const projOp = new ProjectionOperator(node.expressions, evaluators, resultTypes, child.columnMapping, executor.parallelDispatch);
      const childSink = {
        get cancelToken() {
          return currentSink.cancelToken;
        },
        async consume(chunk) {
          if (this.cancelToken?.isCancelled) return;
          const projected = await projOp.process(chunk);
          await currentSink.consume(projected);
        },
        async finalize() {
          if (currentSink.finalize) await currentSink.finalize();
        }
      };
      child.register(graph, currentPipelineId, childSink);
    }
  };
}
async function buildSort(executor, node) {
  const child = await executor.buildPipeline(node.children[0]);
  const keyExtractors = node.orderKeys.map((ok) => ({
    eval: compileExpression(ok.expr, child.columnMapping),
    direction: ok.direction || "ASC"
  }));
  return {
    schema: child.schema,
    columnMapping: child.columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const spillHandle = executor.tempManager.allocate("spill", "sort");
      const sortOp = new SortOperator(keyExtractors, node.limit, node.offset || 0, executor.storageBackend.createSpillManager(spillHandle));
      const sortSink = {
        async consume(chunk) {
          await sortOp.consume(chunk);
        },
        async finalize() {
        }
      };
      const childPipelineId = graph.createPipeline(sortSink);
      child.register(graph, childPipelineId, sortSink);
      graph.addDependency(currentPipelineId, childPipelineId);
      graph.setSource(currentPipelineId, async function* () {
        const resultChunks = await sortOp.finalize();
        for (const chunk of resultChunks) {
          await currentSink.consume(chunk);
          yield chunk;
        }
        if (currentSink.finalize) await currentSink.finalize();
      });
    }
  };
}
async function buildTopN(executor, node) {
  const child = await executor.buildPipeline(node.children[0]);
  const keyExtractors = node.orderKeys.map((ok) => ({
    eval: compileExpression(ok.expr, child.columnMapping),
    direction: ok.direction || "ASC"
  }));
  return {
    schema: child.schema,
    columnMapping: child.columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const spillHandle = executor.tempManager.allocate("spill", "topn");
      const sortOp = new SortOperator(keyExtractors, node.count, node.offset || 0, executor.storageBackend.createSpillManager(spillHandle));
      const sortSink = {
        async consume(chunk) {
          await sortOp.consume(chunk);
        },
        async finalize() {
        }
      };
      const childPipelineId = graph.createPipeline(sortSink);
      child.register(graph, childPipelineId, sortSink);
      graph.addDependency(currentPipelineId, childPipelineId);
      graph.setSource(currentPipelineId, async function* () {
        const resultChunks = await sortOp.finalize();
        for (const chunk of resultChunks) {
          await currentSink.consume(chunk);
          yield chunk;
        }
        if (currentSink.finalize) await currentSink.finalize();
      });
    }
  };
}
async function buildLimit(executor, node) {
  const child = await executor.buildPipeline(node.children[0]);
  const limit = node.count;
  const offset = node.offset || 0;
  return {
    schema: child.schema,
    columnMapping: child.columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const limitOp = new LimitOperator(limit, offset);
      const cancelToken = new CancelToken();
      const childSink = {
        async consume(chunk) {
          if (cancelToken.isCancelled) return;
          await limitOp.consume(chunk);
          const resultChunks = await limitOp.finalize();
          for (const rc of resultChunks) {
            if (rc.size > 0) await currentSink.consume(rc);
          }
          limitOp.chunks = [];
          if (limitOp.done) {
            cancelToken.cancel();
          }
        },
        async finalize() {
          const resultChunks = await limitOp.finalize();
          for (const rc of resultChunks) {
            if (rc.size > 0) await currentSink.consume(rc);
          }
          if (currentSink.finalize) await currentSink.finalize();
        },
        cancelToken
      };
      child.register(graph, currentPipelineId, childSink);
    }
  };
}
async function buildDistinct(executor, node) {
  const child = await executor.buildPipeline(node.children[0]);
  return {
    schema: child.schema,
    columnMapping: child.columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const distinctOp = new DistinctOperator();
      const childSink = {
        async consume(chunk) {
          const result = await distinctOp.process(chunk);
          if (result && result.size > 0) {
            await currentSink.consume(result);
          }
        },
        async finalize() {
          if (currentSink.finalize) await currentSink.finalize();
        }
      };
      child.register(graph, currentPipelineId, childSink);
    }
  };
}
async function buildUnion(executor, node) {
  const left = await executor.buildPipeline(node.children[0]);
  const right = await executor.buildPipeline(node.children[1]);
  return {
    schema: left.schema,
    columnMapping: left.columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      if (!node.all) {
        const unionOp = new UnionOperator(false);
        const dedupSink = {
          async consume(chunk) {
            const result = await unionOp.process(chunk);
            if (result && result.size > 0) {
              await currentSink.consume(result);
            }
          },
          async finalize() {
          }
        };
        const leftPipelineId = graph.createPipeline(dedupSink);
        const rightPipelineId = graph.createPipeline(dedupSink);
        left.register(graph, leftPipelineId, dedupSink);
        right.register(graph, rightPipelineId, dedupSink);
        graph.addDependency(rightPipelineId, leftPipelineId);
        graph.addDependency(currentPipelineId, rightPipelineId);
        graph.setSource(currentPipelineId, async function* () {
          if (currentSink.finalize) await currentSink.finalize();
        });
      } else {
        const leftPipelineId = graph.createPipeline(currentSink);
        const rightPipelineId = graph.createPipeline(currentSink);
        left.register(graph, leftPipelineId, currentSink);
        right.register(graph, rightPipelineId, currentSink);
        graph.addDependency(currentPipelineId, leftPipelineId);
        graph.addDependency(currentPipelineId, rightPipelineId);
        graph.setSource(currentPipelineId, async function* () {
        });
      }
    }
  };
}
async function buildWindow(executor, node) {
  const child = await executor.buildPipeline(node.children[0]);
  const windowExprs = node.windowExprs;
  const windowSchema = [
    ...child.schema,
    ...windowExprs.map((w, i) => ({
      name: `__window_${i}`,
      dataType: executor.normalizeExecType(w.resultType || "FLOAT64"),
      tableAlias: ""
    }))
  ];
  const windowMapping = /* @__PURE__ */ new Map();
  let idx = 0;
  for (const col2 of windowSchema) {
    const key = col2.tableAlias ? `${col2.tableAlias}.${col2.name}`.toUpperCase() : col2.name.toUpperCase();
    windowMapping.set(key, idx);
    if (!windowMapping.has(col2.name.toUpperCase())) {
      windowMapping.set(col2.name.toUpperCase(), idx);
    }
    idx++;
  }
  for (let w = 0; w < windowExprs.length; w++) {
    const wKey = windowExprKey2(windowExprs[w]);
    windowMapping.set(wKey, child.schema.length + w);
  }
  return {
    schema: windowSchema,
    columnMapping: windowMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const childChunks = registerBufferedChild(graph, currentPipelineId, child);
      graph.setSource(currentPipelineId, async function* () {
        const windowOp = new WindowOperator(windowExprs, child.schema, child.columnMapping, compileExpression);
        const resultChunks = await windowOp.execute(childChunks);
        for (const chunk of resultChunks) {
          await currentSink.consume(chunk);
          yield chunk;
        }
        if (currentSink.finalize) await currentSink.finalize();
      });
    }
  };
}
function windowExprKey2(expr2) {
  const name = expr2.name?.toUpperCase() || "WIN";
  const argKey = (expr2.args || []).map((a) => {
    if (a.kind === BoundExprKind.COLUMN_REF) return `${a.tableAlias}.${a.columnName}`.toUpperCase();
    return JSON.stringify(a).slice(0, 30);
  }).join(",");
  const partKey = (expr2.partitionBy || []).map((p) => {
    if (p.kind === BoundExprKind.COLUMN_REF) return `${p.tableAlias}.${p.columnName}`.toUpperCase();
    return "";
  }).join(",");
  return `__WIN__${name}(${argKey})[${partKey}]`;
}

// src/execution/builders/join-builder.js
init_buffer_shim();
init_logical_plan();

// src/execution/join-utils.js
init_buffer_shim();
init_expression_binder();
function splitAnd(expr2) {
  if (!expr2) return [];
  if (expr2.kind === BoundExprKind.BINARY && expr2.op === "AND") {
    return [...splitAnd(expr2.left), ...splitAnd(expr2.right)];
  }
  return [expr2];
}
function hasColumn(mapping, colRef) {
  const fullKey = `${colRef.tableAlias || ""}.${colRef.columnName}`.toUpperCase();
  if (mapping.has(fullKey)) return true;
  const shortKey = colRef.columnName.toUpperCase();
  return mapping.has(shortKey);
}
function findCommonEquiJoinKeys(expr2, leftMapping, rightMapping) {
  if (!expr2) return null;
  if (expr2.kind === BoundExprKind.BINARY && expr2.op === "OR") {
    const leftKeys = findCommonEquiJoinKeys(expr2.left, leftMapping, rightMapping);
    const rightKeys = findCommonEquiJoinKeys(expr2.right, leftMapping, rightMapping);
    if (!leftKeys || !rightKeys) return null;
    if (leftKeys.buildKey.tableAlias === rightKeys.buildKey.tableAlias && leftKeys.buildKey.columnName === rightKeys.buildKey.columnName && leftKeys.probeKey.tableAlias === rightKeys.probeKey.tableAlias && leftKeys.probeKey.columnName === rightKeys.probeKey.columnName) {
      return leftKeys;
    }
    return null;
  }
  if (expr2.kind === BoundExprKind.BINARY && expr2.op === "AND") {
    const leftKeys = findCommonEquiJoinKeys(expr2.left, leftMapping, rightMapping);
    if (leftKeys) return leftKeys;
    return findCommonEquiJoinKeys(expr2.right, leftMapping, rightMapping);
  }
  if (expr2.kind === BoundExprKind.BINARY && expr2.op === "=") {
    if (expr2.left?.kind === BoundExprKind.COLUMN_REF && expr2.right?.kind === BoundExprKind.COLUMN_REF) {
      if (hasColumn(leftMapping, expr2.left) && hasColumn(rightMapping, expr2.right)) {
        return { buildKey: expr2.left, probeKey: expr2.right };
      } else if (hasColumn(leftMapping, expr2.right) && hasColumn(rightMapping, expr2.left)) {
        return { buildKey: expr2.right, probeKey: expr2.left };
      }
    }
  }
  return null;
}
function extractJoinKeys(condition, leftMapping, rightMapping) {
  if (!condition) {
    return { buildKeys: [], probeKeys: [], residualCondition: null };
  }
  const buildKeys = [];
  const probeKeys = [];
  const residualPreds = [];
  const preds = splitAnd(condition);
  for (const pred of preds) {
    let isEquiJoin = false;
    if (pred.kind === BoundExprKind.BINARY && pred.op === "=" && pred.left?.kind === BoundExprKind.COLUMN_REF && pred.right?.kind === BoundExprKind.COLUMN_REF) {
      if (hasColumn(leftMapping, pred.left) && hasColumn(rightMapping, pred.right)) {
        buildKeys.push(pred.left);
        probeKeys.push(pred.right);
        isEquiJoin = true;
      } else if (hasColumn(leftMapping, pred.right) && hasColumn(rightMapping, pred.left)) {
        buildKeys.push(pred.right);
        probeKeys.push(pred.left);
        isEquiJoin = true;
      }
    }
    if (!isEquiJoin) {
      residualPreds.push(pred);
    }
  }
  if (buildKeys.length === 0) {
    const commonKeys = findCommonEquiJoinKeys(condition, leftMapping, rightMapping);
    if (commonKeys && commonKeys.buildKey) {
      buildKeys.push(commonKeys.buildKey);
      probeKeys.push(commonKeys.probeKey);
      residualPreds.length = 0;
      residualPreds.push(condition);
    } else {
      buildKeys.push({ kind: BoundExprKind.LITERAL, value: 1 });
      probeKeys.push({ kind: BoundExprKind.LITERAL, value: 1 });
    }
  }
  let residualCondition = null;
  if (residualPreds.length > 0) {
    residualCondition = residualPreds.reduce((acc, p) => ({
      kind: BoundExprKind.BINARY,
      op: "AND",
      left: acc,
      right: p,
      resultType: "BOOLEAN"
    }));
  }
  return { buildKeys, probeKeys, residualCondition };
}

// src/execution/operators/hash-join.js
init_buffer_shim();
init_column();
init_chunk();
init_logical_plan();
init_config();

// src/execution/operators/join-core.js
init_buffer_shim();
init_column();
init_chunk();
init_logical_plan();
init_sab_arena();
function joinKeyOf(extractors, chunk, rowIdx) {
  if (extractors.length === 1) {
    const val = extractors[0](chunk, rowIdx);
    if (val === null || val === void 0) return null;
    return typeof val === "bigint" ? Number(val) : val;
  }
  const parts = new Array(extractors.length);
  for (let i = 0; i < extractors.length; i++) {
    const val = extractors[i](chunk, rowIdx);
    if (val === null || val === void 0) return null;
    parts[i] = typeof val === "bigint" ? Number(val) : val;
  }
  return parts.join("|");
}
function createCombinedRowAdapter(totalCols) {
  const columns = new Array(totalCols);
  const adapter = {
    row: null,
    columns,
    setRow(r) {
      this.row = r;
    }
  };
  for (let c = 0; c < totalCols; c++) {
    columns[c] = { get: () => adapter.row[c] };
  }
  return adapter;
}
function probeJoinRows(items, lookup, opts) {
  const { joinType, buildColCount, probeColCount, conditionEvaluator, hasNullKey, onMatched } = opts;
  const adapter = conditionEvaluator ? createCombinedRowAdapter(buildColCount + probeColCount) : null;
  const resultRows = [];
  for (const item of items) {
    const { row: pRow, key } = item;
    if (key === null) {
      if (joinType === JoinType.LEFT || joinType === JoinType.FULL || joinType === JoinType.ANTI || joinType === JoinType.SINGLE) {
        resultRows.push(new Array(buildColCount).fill(null).concat(pRow));
      }
      continue;
    }
    const bucket = lookup(key);
    let matched = false;
    if (bucket) {
      for (const buildItem of bucket) {
        const bRow = buildItem.row;
        if (adapter) {
          const combined = bRow.concat(pRow);
          adapter.setRow(combined);
          if (!conditionEvaluator(adapter, 0)) continue;
        }
        matched = true;
        if (onMatched) onMatched(buildItem);
        if (joinType === JoinType.SEMI) {
          break;
        } else if (joinType === JoinType.ANTI) {
          break;
        } else if (joinType === JoinType.SINGLE) {
          resultRows.push(bRow.concat(pRow));
          break;
        } else if (joinType === JoinType.MARK) {
          break;
        } else {
          resultRows.push(bRow.concat(pRow));
        }
      }
    }
    if (!matched) {
      if (joinType === JoinType.LEFT || joinType === JoinType.FULL || joinType === JoinType.SINGLE) {
        resultRows.push(new Array(buildColCount).fill(null).concat(pRow));
      } else if (joinType === JoinType.ANTI) {
        resultRows.push(pRow);
      } else if (joinType === JoinType.MARK) {
        resultRows.push(pRow.concat([hasNullKey ? null : false]));
      }
    } else {
      if (joinType === JoinType.SEMI) {
        resultRows.push(pRow);
      } else if (joinType === JoinType.MARK) {
        resultRows.push(pRow.concat([true]));
      }
    }
  }
  return resultRows;
}
function buildJoinOutputChunk(rows, { joinType, buildColCount, buildSchema, probeSchema }, allocator = heapAllocator) {
  if (rows.length === 0) {
    return new DataChunk([], 0);
  }
  const isSemiAnti = joinType === JoinType.SEMI || joinType === JoinType.ANTI;
  const isMark = joinType === JoinType.MARK;
  const colCount = rows[0].length;
  const cols = [];
  for (let c = 0; c < colCount; c++) {
    const firstVal = rows.find((r) => r[c] !== null)?.[c];
    let dt = "VARCHAR";
    if (firstVal !== void 0) {
      dt = typeof firstVal === "bigint" ? "DECIMAL" : typeof firstVal === "number" ? Number.isInteger(firstVal) ? "INT32" : "FLOAT64" : typeof firstVal === "boolean" ? "BOOLEAN" : "VARCHAR";
    }
    let finalDt = dt;
    if (isSemiAnti) {
      finalDt = probeSchema?.[c] || dt;
    } else if (isMark && c === colCount - 1) {
      finalDt = "BOOLEAN";
    } else {
      if (c < buildColCount) {
        finalDt = buildSchema?.[c] || dt;
      } else {
        finalDt = probeSchema?.[c - buildColCount] || dt;
      }
    }
    const col2 = new Column(finalDt, rows.length, allocator);
    for (let r = 0; r < rows.length; r++) {
      col2.set(r, rows[r][c]);
    }
    col2.length = rows.length;
    cols.push(col2);
  }
  return new DataChunk(cols, rows.length);
}

// src/execution/operators/hash-join.js
function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}
function getPartition(keyStr) {
  return hashString(keyStr) % Config.hashJoinPartitions;
}
var HashJoinBuild = class {
  constructor(keyExtractors, joinType, uniqueKeys, spillManager) {
    this.keyExtractors = keyExtractors;
    this.joinType = joinType || JoinType.INNER;
    this.uniqueKeys = !!uniqueKeys;
    this.hashTable = /* @__PURE__ */ new Map();
    this.buildSchema = null;
    this.hasNullKey = false;
    this.spillManager = spillManager;
    this.partitions = Array.from({ length: Config.hashJoinPartitions }, () => ({
      rows: [],
      spilled: false
    }));
    this.totalRowsInRAM = 0;
    this.matchedSet = /* @__PURE__ */ new Set();
  }
  async init() {
  }
  async consume(chunk) {
    if (!this.buildSchema) {
      this.buildSchema = chunk.columns.map((c) => c.dataType);
    }
    const flat = chunk.selectionVector ? chunk.flatten() : chunk;
    const chunkRows = new Array(flat.size);
    for (let i = 0; i < flat.size; i++) {
      const row = new Array(flat.columns.length);
      for (let c = 0; c < flat.columns.length; c++) {
        row[c] = flat.columns[c].get(i);
      }
      chunkRows[i] = row;
    }
    for (let i = 0; i < flat.size; i++) {
      const key = this.buildKey(flat, i);
      if (key === null) {
        this.hasNullKey = true;
        continue;
      }
      const pIdx = getPartition(key);
      const part = this.partitions[pIdx];
      part.rows.push({ row: chunkRows[i], key });
      if (!part.spilled) {
        this.totalRowsInRAM++;
      }
      if (part.spilled && part.rows.length >= Config.flushBatchSize) {
        await this.flushPartition(pIdx);
      }
    }
    if (this.totalRowsInRAM > Config.memoryLimit) {
      let maxPart = -1;
      let maxRows = 0;
      for (let i = 0; i < Config.hashJoinPartitions; i++) {
        if (!this.partitions[i].spilled && this.partitions[i].rows.length > maxRows) {
          maxRows = this.partitions[i].rows.length;
          maxPart = i;
        }
      }
      if (maxPart !== -1) {
        this.partitions[maxPart].spilled = true;
        this.totalRowsInRAM -= this.partitions[maxPart].rows.length;
        await this.flushPartition(maxPart);
      }
    }
  }
  async flushPartition(pIdx) {
    const part = this.partitions[pIdx];
    if (part.rows.length === 0) return;
    const chunk = this.rowsToChunk(part.rows.map((r) => r.row));
    await this.spillManager.appendChunk(`build_${pIdx}`, chunk);
    part.rows = [];
  }
  rowsToChunk(rows) {
    if (rows.length === 0) return new DataChunk([], 0);
    const colCount = rows[0].length;
    const columns = new Array(colCount);
    for (let c = 0; c < colCount; c++) {
      const col2 = new Column(this.buildSchema?.[c] || "VARCHAR", rows.length);
      for (let r = 0; r < rows.length; r++) {
        col2.set(r, rows[r][c]);
      }
      col2.length = rows.length;
      columns[c] = col2;
    }
    return new DataChunk(columns, rows.length);
  }
  async finalize() {
    for (let i = 0; i < Config.hashJoinPartitions; i++) {
      const part = this.partitions[i];
      if (part.spilled && part.rows.length > 0) {
        await this.flushPartition(i);
      }
      if (!part.spilled) {
        for (let r = 0; r < part.rows.length; r++) {
          const item = part.rows[r];
          let bucket = this.hashTable.get(item.key);
          if (this.uniqueKeys && bucket) continue;
          if (!bucket) {
            bucket = [];
            this.hashTable.set(item.key, bucket);
          }
          bucket.push({ row: item.row, pIdx: i, rIdx: r });
        }
      }
    }
  }
  markMatched(packed) {
    this.matchedSet.add(`${packed.pIdx}_${packed.rIdx}`);
  }
  emitUnmatched(probeColCount) {
    const rows = [];
    for (let i = 0; i < Config.hashJoinPartitions; i++) {
      const part = this.partitions[i];
      if (!part.spilled) {
        for (let r = 0; r < part.rows.length; r++) {
          if (!this.matchedSet.has(`${i}_${r}`)) {
            const outRow = [...part.rows[r].row];
            for (let c = 0; c < probeColCount; c++) outRow.push(null);
            rows.push(outRow);
          }
        }
      }
    }
    return rows;
  }
  probe(key) {
    return this.hashTable.get(key) || null;
  }
  buildKey(chunk, rowIdx) {
    return joinKeyOf(this.keyExtractors, chunk, rowIdx);
  }
};
var HashJoinProbe = class {
  constructor(buildSide, probeKeyExtractors, buildColCount, probeColCount, joinType = JoinType.INNER, conditionEvaluator = null) {
    this.buildSide = buildSide;
    this.probeKeyExtractors = probeKeyExtractors;
    this.buildColCount = buildColCount;
    this.probeColCount = probeColCount;
    this.joinType = joinType;
    this.conditionEvaluator = conditionEvaluator;
    this.spillBuffers = Array.from({ length: Config.hashJoinPartitions }, () => []);
    this.probeSchema = null;
  }
  async init() {
  }
  async process(probeChunk) {
    if (!this.probeSchema) {
      this.probeSchema = probeChunk.columns.map((c) => c.dataType);
    }
    const flat = probeChunk.selectionVector ? probeChunk.flatten() : probeChunk;
    const inMemoryRows = [];
    for (let i = 0; i < flat.size; i++) {
      const key = this.extractProbeKey(flat, i);
      const row = new Array(flat.columns.length);
      for (let c = 0; c < flat.columns.length; c++) {
        row[c] = flat.columns[c].get(i);
      }
      if (key === null) {
        inMemoryRows.push({ row, key: null });
        continue;
      }
      const pIdx = getPartition(key);
      if (this.buildSide.partitions[pIdx].spilled) {
        this.spillBuffers[pIdx].push({ row, key });
        if (this.spillBuffers[pIdx].length >= Config.flushBatchSize) {
          await this.flushProbePartition(pIdx);
        }
      } else {
        inMemoryRows.push({ row, key });
      }
    }
    if (inMemoryRows.length > 0) {
      return this.executeInMemoryJoin(inMemoryRows);
    }
    return null;
  }
  async flushProbePartition(pIdx) {
    const buffer = this.spillBuffers[pIdx];
    if (buffer.length === 0) return;
    const chunk = this.rowsToProbeChunk(buffer.map((r) => r.row));
    await this.buildSide.spillManager.appendChunk(`probe_${pIdx}`, chunk);
    this.spillBuffers[pIdx] = [];
  }
  rowsToProbeChunk(rows) {
    if (rows.length === 0) return new DataChunk([], 0);
    const colCount = rows[0].length;
    const columns = new Array(colCount);
    for (let c = 0; c < colCount; c++) {
      const col2 = new Column(this.probeSchema?.[c] || "VARCHAR", rows.length);
      for (let r = 0; r < rows.length; r++) {
        col2.set(r, rows[r][c]);
      }
      col2.length = rows.length;
      columns[c] = col2;
    }
    return new DataChunk(columns, rows.length);
  }
  executeInMemoryJoin(probeItems) {
    const resultRows = probeJoinRows(probeItems, (key) => this.buildSide.probe(key), {
      joinType: this.joinType,
      buildColCount: this.buildColCount,
      probeColCount: this.probeColCount,
      conditionEvaluator: this.conditionEvaluator,
      hasNullKey: this.buildSide.hasNullKey,
      onMatched: (buildItem) => this.buildSide.markMatched(buildItem)
    });
    return this.buildOutputChunk(resultRows);
  }
  async finalize(sink) {
    for (let i = 0; i < Config.hashJoinPartitions; i++) {
      if (this.spillBuffers[i].length > 0) {
        await this.flushProbePartition(i);
      }
    }
    for (let i = 0; i < Config.hashJoinPartitions; i++) {
      const part = this.buildSide.partitions[i];
      if (!part.spilled) continue;
      this.buildSide.hashTable.clear();
      part.rows = [];
      const buildIter = this.buildSide.spillManager.readChunks(`build_${i}`);
      for await (const bChunk of buildIter) {
        for (let r = 0; r < bChunk.size; r++) {
          const key = this.buildSide.buildKey(bChunk, r);
          const row = new Array(bChunk.columns.length);
          for (let c = 0; c < bChunk.columns.length; c++) row[c] = bChunk.columns[c].get(r);
          const rIdx = part.rows.length;
          part.rows.push({ row, key });
          let bucket = this.buildSide.hashTable.get(key);
          if (this.buildSide.uniqueKeys && bucket) continue;
          if (!bucket) {
            bucket = [];
            this.buildSide.hashTable.set(key, bucket);
          }
          bucket.push({ row, pIdx: i, rIdx });
        }
      }
      const probeIter = this.buildSide.spillManager.readChunks(`probe_${i}`);
      for await (const pChunk of probeIter) {
        const pItems = [];
        for (let r = 0; r < pChunk.size; r++) {
          const key = this.extractProbeKey(pChunk, r);
          const row = new Array(pChunk.columns.length);
          for (let c = 0; c < pChunk.columns.length; c++) row[c] = pChunk.columns[c].get(r);
          pItems.push({ row, key });
        }
        const result = this.executeInMemoryJoin(pItems);
        if (result && result.size > 0) {
          await sink.consume(result);
        }
      }
    }
    await this.buildSide.spillManager.clearAll();
  }
  extractProbeKey(chunk, rowIdx) {
    return joinKeyOf(this.probeKeyExtractors, chunk, rowIdx);
  }
  buildOutputChunk(rows) {
    return buildJoinOutputChunk(rows, {
      joinType: this.joinType,
      buildColCount: this.buildColCount,
      buildSchema: this.buildSide.buildSchema,
      probeSchema: this.probeSchema
    });
  }
};

// src/execution/operators/merge-join.js
init_buffer_shim();
init_chunk();
init_column();
init_logical_plan();
init_data_type();
function isNullKey(key) {
  if (key === null || key === void 0) return true;
  return Array.isArray(key) && key.some((k) => k === null || k === void 0);
}
var MergeJoinOperator = class {
  constructor(buildChunks2, probeChunks, buildKeyExtractors2, probeKeyExtractors, buildColCount, probeColCount, joinType = JoinType.INNER, conditionEvaluator = null) {
    this.buildChunks = buildChunks2;
    this.probeChunks = probeChunks;
    this.buildKeyExtractors = buildKeyExtractors2;
    this.probeKeyExtractors = probeKeyExtractors;
    this.buildColCount = buildColCount;
    this.probeColCount = probeColCount;
    this.joinType = joinType;
    this.conditionEvaluator = conditionEvaluator;
  }
  async execute() {
    const isSemiAnti = this.joinType === JoinType.SEMI || this.joinType === JoinType.ANTI;
    const isMark = this.joinType === JoinType.MARK;
    const buildAll = this._flattenAndExtractKeys(this.buildChunks, this.buildKeyExtractors);
    const probeAll = this._flattenAndExtractKeys(this.probeChunks, this.probeKeyExtractors);
    const buildRows = buildAll.filter((r) => !isNullKey(r.key));
    const probeRows = probeAll.filter((r) => !isNullKey(r.key));
    const buildNull = buildAll.filter((r) => isNullKey(r.key));
    const probeNull = probeAll.filter((r) => isNullKey(r.key));
    const markUnmatched = buildNull.length > 0 ? null : false;
    buildRows.sort((a, b2) => this._compareKeys(a.key, b2.key));
    probeRows.sort((a, b2) => this._compareKeys(a.key, b2.key));
    const outputRows = [];
    const adapter = this.conditionEvaluator ? this.createAdapter() : null;
    let b = 0;
    let p = 0;
    while (b < buildRows.length && p < probeRows.length) {
      const bRow = buildRows[b];
      const pRow = probeRows[p];
      const cmp = this._compareKeys(bRow.key, pRow.key);
      if (cmp < 0) {
        if (this.joinType === JoinType.LEFT || this.joinType === JoinType.FULL) {
          outputRows.push(this._combineRowWithNulls(bRow, true));
        }
        b++;
      } else if (cmp > 0) {
        if (isSemiAnti && this.joinType === JoinType.ANTI) {
          outputRows.push(this._extractProbeRow(pRow));
        } else if (isMark) {
          outputRows.push(this._extractProbeRow(pRow).concat([markUnmatched]));
        } else if (this.joinType === JoinType.RIGHT || this.joinType === JoinType.FULL) {
          outputRows.push(this._combineRowWithNulls(pRow, false));
        }
        p++;
      } else {
        let bEnd = b;
        while (bEnd < buildRows.length && this._compareKeys(bRow.key, buildRows[bEnd].key) === 0) {
          bEnd++;
        }
        let pEnd = p;
        while (pEnd < probeRows.length && this._compareKeys(pRow.key, probeRows[pEnd].key) === 0) {
          pEnd++;
        }
        if (isSemiAnti || isMark) {
          for (let j = p; j < pEnd; j++) {
            let matched = false;
            for (let i = b; i < bEnd; i++) {
              if (adapter) {
                const row = this._combineRow(buildRows[i], probeRows[j]);
                adapter.setRow(row);
                if (!this.conditionEvaluator(adapter, 0)) continue;
              }
              matched = true;
              break;
            }
            if (this.joinType === JoinType.SEMI && matched) {
              outputRows.push(this._extractProbeRow(probeRows[j]));
            } else if (this.joinType === JoinType.ANTI && !matched) {
              outputRows.push(this._extractProbeRow(probeRows[j]));
            } else if (isMark) {
              outputRows.push(this._extractProbeRow(probeRows[j]).concat([matched ? true : markUnmatched]));
            }
          }
        } else {
          for (let i = b; i < bEnd; i++) {
            let matchedAny = false;
            for (let j = p; j < pEnd; j++) {
              const row = this._combineRow(buildRows[i], probeRows[j]);
              if (adapter) {
                adapter.setRow(row);
                if (!this.conditionEvaluator(adapter, 0)) continue;
              }
              outputRows.push(row);
              matchedAny = true;
            }
            if (!matchedAny && (this.joinType === JoinType.LEFT || this.joinType === JoinType.FULL)) {
              outputRows.push(this._combineRowWithNulls(buildRows[i], true));
            }
          }
        }
        b = bEnd;
        p = pEnd;
      }
    }
    while (b < buildRows.length) {
      if (this.joinType === JoinType.LEFT || this.joinType === JoinType.FULL) {
        outputRows.push(this._combineRowWithNulls(buildRows[b], true));
      }
      b++;
    }
    while (p < probeRows.length) {
      if (isSemiAnti && this.joinType === JoinType.ANTI) {
        outputRows.push(this._extractProbeRow(probeRows[p]));
      } else if (isMark) {
        outputRows.push(this._extractProbeRow(probeRows[p]).concat([markUnmatched]));
      } else if (this.joinType === JoinType.RIGHT || this.joinType === JoinType.FULL) {
        outputRows.push(this._combineRowWithNulls(probeRows[p], false));
      }
      p++;
    }
    for (const bRow of buildNull) {
      if (this.joinType === JoinType.LEFT || this.joinType === JoinType.FULL) {
        outputRows.push(this._combineRowWithNulls(bRow, true));
      }
    }
    for (const pRow of probeNull) {
      if (isSemiAnti && this.joinType === JoinType.ANTI) {
        outputRows.push(this._extractProbeRow(pRow));
      } else if (isMark) {
        outputRows.push(this._extractProbeRow(pRow).concat([null]));
      } else if (this.joinType === JoinType.RIGHT || this.joinType === JoinType.FULL) {
        outputRows.push(this._combineRowWithNulls(pRow, false));
      }
    }
    if (outputRows.length === 0) return [];
    return [this._buildOutputChunk(outputRows)];
  }
  _flattenAndExtractKeys(chunks, extractors) {
    const rows = [];
    for (const chunk of chunks) {
      for (let i = 0; i < chunk.size; i++) {
        const idx = chunk.activeRowIndex(i);
        const key = extractors.length === 1 ? extractors[0](chunk, idx) : extractors.map((fn) => fn(chunk, idx));
        rows.push({ chunk, idx, key });
      }
    }
    return rows;
  }
  _compareKeys(k1, k2) {
    if (Array.isArray(k1)) {
      for (let i = 0; i < k1.length; i++) {
        const c12 = typeof k1[i] === "bigint" ? Number(k1[i]) : k1[i];
        const c22 = typeof k2[i] === "bigint" ? Number(k2[i]) : k2[i];
        if (c12 < c22) return -1;
        if (c12 > c22) return 1;
      }
      return 0;
    }
    const c1 = typeof k1 === "bigint" ? Number(k1) : k1;
    const c2 = typeof k2 === "bigint" ? Number(k2) : k2;
    if (c1 < c2) return -1;
    if (c1 > c2) return 1;
    return 0;
  }
  _combineRow(bRow, pRow) {
    const row = [];
    for (let c = 0; c < bRow.chunk.columns.length; c++) {
      row.push(bRow.chunk.columns[c].get(bRow.idx));
    }
    for (let c = 0; c < pRow.chunk.columns.length; c++) {
      row.push(pRow.chunk.columns[c].get(pRow.idx));
    }
    return row;
  }
  _extractProbeRow(rowObj) {
    const row = [];
    for (let c = 0; c < rowObj.chunk.columns.length; c++) {
      row.push(rowObj.chunk.columns[c].get(rowObj.idx));
    }
    return row;
  }
  _combineRowWithNulls(rowObj, isBuild) {
    const row = [];
    if (isBuild) {
      for (let c = 0; c < rowObj.chunk.columns.length; c++) {
        row.push(rowObj.chunk.columns[c].get(rowObj.idx));
      }
      for (let c = 0; c < this.probeColCount; c++) row.push(null);
    } else {
      for (let c = 0; c < this.buildColCount; c++) row.push(null);
      for (let c = 0; c < rowObj.chunk.columns.length; c++) {
        row.push(rowObj.chunk.columns[c].get(rowObj.idx));
      }
    }
    return row;
  }
  _buildOutputChunk(outputRows) {
    if (outputRows.length === 0) return null;
    const isSemiAnti = this.joinType === JoinType.SEMI || this.joinType === JoinType.ANTI;
    const isMark = this.joinType === JoinType.MARK;
    const colCount = isSemiAnti ? this.probeColCount : isMark ? this.probeColCount + 1 : this.buildColCount + this.probeColCount;
    const columns = [];
    for (let c = 0; c < colCount; c++) {
      let dt = DataType.VARCHAR;
      if (isSemiAnti) {
        if (this.probeChunks.length > 0 && c < this.probeChunks[0].columns.length) dt = this.probeChunks[0].columns[c].dataType;
      } else if (isMark) {
        if (c < this.probeColCount && this.probeChunks.length > 0 && c < this.probeChunks[0].columns.length) dt = this.probeChunks[0].columns[c].dataType;
        else if (c === this.probeColCount) dt = DataType.BOOLEAN;
      } else {
        if (c < this.buildColCount && this.buildChunks.length > 0) dt = this.buildChunks[0].columns[c].dataType;
        else if (c >= this.buildColCount && this.probeChunks.length > 0) dt = this.probeChunks[0].columns[c - this.buildColCount].dataType;
      }
      columns.push(new Column(dt, outputRows.length));
    }
    for (let r = 0; r < outputRows.length; r++) {
      const row = outputRows[r];
      for (let c = 0; c < colCount; c++) {
        columns[c].set(r, row[c]);
      }
    }
    for (const col2 of columns) col2.length = outputRows.length;
    return new DataChunk(columns, outputRows.length);
  }
  createAdapter() {
    const totalCols = this.buildColCount + this.probeColCount;
    const columns = new Array(totalCols);
    const adapter = {
      row: null,
      columns,
      setRow(r) {
        this.row = r;
      }
    };
    for (let c = 0; c < totalCols; c++) {
      columns[c] = { get: () => adapter.row[c] };
    }
    return adapter;
  }
};

// src/execution/operators/nested-loop-join.js
init_buffer_shim();
init_chunk();
init_column();
init_logical_plan();
var NestedLoopJoinOperator = class {
  constructor(outerChunks, innerChunks, outerColCount, innerColCount, joinType = JoinType.INNER, conditionEvaluator = null) {
    this.outerChunks = outerChunks;
    this.innerChunks = innerChunks;
    this.outerColCount = outerColCount;
    this.innerColCount = innerColCount;
    this.joinType = joinType;
    this.conditionEvaluator = conditionEvaluator;
  }
  async execute() {
    const outerRows = this._flattenChunks(this.outerChunks);
    const innerRows = this._flattenChunks(this.innerChunks);
    const outputRows = [];
    const adapter = this.conditionEvaluator ? this._createAdapter() : null;
    const innerMatched = this.joinType === JoinType.FULL || this.joinType === JoinType.RIGHT ? new Uint8Array(innerRows.length) : null;
    for (let o = 0; o < outerRows.length; o++) {
      const oRow = outerRows[o];
      let matched = false;
      for (let i = 0; i < innerRows.length; i++) {
        const iRow = innerRows[i];
        const combined = this._combineRow(oRow, iRow);
        if (adapter) {
          adapter.setRow(combined);
          if (!this.conditionEvaluator(adapter, 0)) continue;
        }
        matched = true;
        if (innerMatched) innerMatched[i] = 1;
        if (this.joinType === JoinType.SEMI) {
          outputRows.push(this._extractOuter(oRow));
          break;
        }
        if (this.joinType === JoinType.ANTI) break;
        if (this.joinType === JoinType.MARK) break;
        outputRows.push(combined);
      }
      if (!matched) {
        if (this.joinType === JoinType.LEFT || this.joinType === JoinType.FULL) {
          outputRows.push(this._outerWithNulls(oRow));
        } else if (this.joinType === JoinType.ANTI) {
          outputRows.push(this._extractOuter(oRow));
        } else if (this.joinType === JoinType.MARK) {
          outputRows.push(this._outerWithMark(oRow, false));
        }
      } else if (this.joinType === JoinType.MARK) {
        outputRows.push(this._outerWithMark(oRow, true));
      }
    }
    if (innerMatched) {
      for (let i = 0; i < innerRows.length; i++) {
        if (!innerMatched[i]) {
          outputRows.push(this._innerWithNulls(innerRows[i]));
        }
      }
    }
    if (outputRows.length === 0) return [];
    return [this._buildOutputChunk(outputRows)];
  }
  _flattenChunks(chunks) {
    const rows = [];
    for (const chunk of chunks) {
      for (let i = 0; i < chunk.size; i++) {
        const idx = chunk.activeRowIndex(i);
        const row = new Array(chunk.columns.length);
        for (let c = 0; c < chunk.columns.length; c++) {
          row[c] = chunk.columns[c].get(idx);
        }
        rows.push(row);
      }
    }
    return rows;
  }
  _combineRow(outerRow, innerRow) {
    const row = new Array(this.outerColCount + this.innerColCount);
    for (let c = 0; c < this.outerColCount; c++) row[c] = outerRow[c];
    for (let c = 0; c < this.innerColCount; c++) row[this.outerColCount + c] = innerRow[c];
    return row;
  }
  _extractOuter(outerRow) {
    return outerRow.slice(0, this.outerColCount);
  }
  _outerWithNulls(outerRow) {
    const row = new Array(this.outerColCount + this.innerColCount);
    for (let c = 0; c < this.outerColCount; c++) row[c] = outerRow[c];
    for (let c = 0; c < this.innerColCount; c++) row[this.outerColCount + c] = null;
    return row;
  }
  _innerWithNulls(innerRow) {
    const row = new Array(this.outerColCount + this.innerColCount);
    for (let c = 0; c < this.outerColCount; c++) row[c] = null;
    for (let c = 0; c < this.innerColCount; c++) row[this.outerColCount + c] = innerRow[c];
    return row;
  }
  _outerWithMark(outerRow, markValue) {
    const row = new Array(this.outerColCount + 1);
    for (let c = 0; c < this.outerColCount; c++) row[c] = outerRow[c];
    row[this.outerColCount] = markValue;
    return row;
  }
  _buildOutputChunk(rows) {
    const colCount = rows[0].length;
    const columns = new Array(colCount);
    for (let c = 0; c < colCount; c++) {
      let dt = "VARCHAR";
      if (this.joinType === JoinType.MARK && c === colCount - 1) {
        dt = "BOOLEAN";
      } else if (this.joinType === JoinType.SEMI || this.joinType === JoinType.ANTI) {
        if (c < this.outerColCount && this.outerChunks.length > 0 && this.outerChunks[0].columns[c]) {
          dt = this.outerChunks[0].columns[c].dataType;
        }
      } else {
        if (c < this.outerColCount && this.outerChunks.length > 0 && this.outerChunks[0].columns[c]) {
          dt = this.outerChunks[0].columns[c].dataType;
        } else if (c >= this.outerColCount && this.innerChunks.length > 0) {
          const innerIdx = c - this.outerColCount;
          if (this.innerChunks[0].columns[innerIdx]) {
            dt = this.innerChunks[0].columns[innerIdx].dataType;
          }
        }
      }
      const col2 = new Column(dt, rows.length);
      for (let r = 0; r < rows.length; r++) {
        col2.set(r, rows[r][c]);
      }
      col2.length = rows.length;
      columns[c] = col2;
    }
    return new DataChunk(columns, rows.length);
  }
  _createAdapter() {
    const totalCols = this.outerColCount + this.innerColCount;
    const columns = new Array(totalCols);
    const adapter = {
      row: null,
      columns,
      setRow(r) {
        this.row = r;
      }
    };
    for (let c = 0; c < totalCols; c++) {
      columns[c] = { get: () => adapter.row[c] };
    }
    return adapter;
  }
};

// src/execution/builders/join-builder.js
init_config();
async function buildJoin(executor, node) {
  const left = await executor.buildPipeline(node.children[0]);
  const right = await executor.buildPipeline(node.children[1]);
  const isSemiAnti = node.joinType === JoinType.SEMI || node.joinType === JoinType.ANTI;
  const isMark = node.joinType === JoinType.MARK;
  let buildInput, probeInput, buildNode, probeNode;
  if (isSemiAnti || isMark || node._buildSide === "right") {
    buildInput = right;
    probeInput = left;
    buildNode = node.children[1];
    probeNode = node.children[0];
  } else {
    buildInput = left;
    probeInput = right;
    buildNode = node.children[0];
    probeNode = node.children[1];
  }
  const combinedSchema = [...buildInput.schema, ...probeInput.schema];
  const combinedMapping = combinedMappingOf(buildInput.schema, probeInput.schema);
  const { buildKeys, probeKeys, residualCondition } = extractJoinKeys(
    node.condition,
    buildInput.columnMapping,
    probeInput.columnMapping
  );
  const conditionEvaluator = residualCondition ? compileExpression(residualCondition, combinedMapping) : null;
  const markSchema = isMark ? [...left.schema, { name: node.markColumn || "__mark", dataType: "BOOLEAN", tableAlias: "" }] : null;
  const resultSchema = isSemiAnti ? left.schema : isMark ? markSchema : combinedSchema;
  const resultMapping = isSemiAnti ? left.columnMapping : isMark ? executor.buildSchemaMapping(markSchema, "") : combinedMapping;
  if (node.physicalStrategy === PhysicalStrategy.MERGE) {
    return buildMergeJoin(node, {
      left,
      right,
      buildInput,
      probeInput,
      buildKeys,
      probeKeys,
      conditionEvaluator,
      resultSchema,
      resultMapping
    });
  }
  if (node.physicalStrategy === PhysicalStrategy.NESTED_LOOP) {
    return buildNestedLoopJoin(node, {
      left,
      right,
      buildInput,
      probeInput,
      isSemiAnti,
      isMark,
      markSchema
    });
  }
  const joinSpillHandle = executor.tempManager.allocate("spill", "join");
  const makeBuildSide = () => new HashJoinBuild(
    buildKeys.map((k) => compileExpression(k, buildInput.columnMapping)),
    node.joinType,
    !!node._dedupeBuild && !conditionEvaluator,
    executor.storageBackend.createSpillManager(joinSpillHandle)
  );
  const makeProbeOp = (buildSide) => new HashJoinProbe(
    buildSide,
    probeKeys.map((k) => compileExpression(k, probeInput.columnMapping)),
    buildInput.schema.length,
    probeInput.schema.length,
    node.joinType,
    conditionEvaluator
  );
  const serialCompiled = {
    schema: resultSchema,
    columnMapping: resultMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const buildSide = makeBuildSide();
      const buildSink = {
        async consume(chunk) {
          await buildSide.consume(chunk);
        },
        async finalize() {
          await buildSide.finalize();
        }
      };
      const buildPipelineId = graph.createPipeline(buildSink);
      buildInput.register(graph, buildPipelineId, buildSink);
      const probeOp = makeProbeOp(buildSide);
      const probeSink = {
        get cancelToken() {
          return currentSink.cancelToken;
        },
        async consume(chunk) {
          if (this.cancelToken?.isCancelled) return;
          const result = await probeOp.process(chunk);
          if (result && result.size > 0) {
            await currentSink.consume(result);
          }
        },
        async finalize() {
          const buildIsPreservedSide = node.joinType === JoinType.FULL || node.joinType === JoinType.LEFT && buildNode === node.children[0];
          if (buildIsPreservedSide) {
            const unmatchedRows = buildSide.emitUnmatched(probeInput.schema.length);
            if (unmatchedRows.length > 0) {
              await currentSink.consume(probeOp.buildOutputChunk(unmatchedRows));
            }
          }
          if (probeOp.finalize) {
            await probeOp.finalize(currentSink);
          }
          if (currentSink.finalize) await currentSink.finalize();
        }
      };
      graph.addDependency(currentPipelineId, buildPipelineId);
      probeInput.register(graph, currentPipelineId, probeSink);
    }
  };
  const parallelJoin = executor._prepareParallelJoin(
    node,
    buildInput,
    probeInput,
    buildNode,
    probeNode,
    buildKeys,
    probeKeys,
    residualCondition,
    combinedMapping
  );
  if (!parallelJoin) return serialCompiled;
  return {
    schema: resultSchema,
    columnMapping: resultMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const registerSide = (side, input) => side.storage ? null : registerBufferedChild(graph, currentPipelineId, input);
      const bufferedBuild = registerSide(parallelJoin.buildSide, buildInput);
      const bufferedProbe = registerSide(parallelJoin.probeSide, probeInput);
      graph.setSource(currentPipelineId, async function* () {
        const collect = async (side, buffered) => {
          if (!side.storage) return buffered;
          const chunks = [];
          for await (const chunk of side.storage.scan()) chunks.push(chunk);
          return chunks;
        };
        const buildChunks2 = await collect(parallelJoin.buildSide, bufferedBuild);
        const probeChunks = await collect(parallelJoin.probeSide, bufferedProbe);
        const countRows = (chunks) => chunks.reduce((sum2, c) => sum2 + c.size, 0);
        const buildRows = countRows(buildChunks2);
        const probeRows = countRows(probeChunks);
        const eligible = buildRows + probeRows >= Config.parallelJoinThreshold && buildRows <= Config.memoryLimit;
        const emitSerial = async function* () {
          const bothBuffered = bufferedBuild !== null && bufferedProbe !== null;
          const resultChunks = bothBuffered ? await executor._runBufferedSerialJoin(makeBuildSide, makeProbeOp, buildChunks2, probeChunks, node, probeInput.schema.length) : await executor._executeSubPipeline(serialCompiled);
          for (const chunk of resultChunks) {
            if (!chunk || chunk.size === 0) continue;
            await currentSink.consume(chunk);
            yield chunk;
          }
        };
        if (!eligible) {
          yield* emitSerial();
          if (currentSink.finalize) await currentSink.finalize();
          return;
        }
        const typesOf = (side, chunks) => {
          if (side.storage) return side.spec.schema.map((col2) => col2.dataType);
          const first = chunks.find((c) => c.size > 0);
          if (first) return first.columns.map((col2) => col2.dataType);
          return side.spec.schema.map((col2) => col2.dataType);
        };
        const outputTypes = {
          build: typesOf(parallelJoin.buildSide, buildChunks2),
          probe: typesOf(parallelJoin.probeSide, probeChunks)
        };
        let emitted = false;
        try {
          const stream = executor.fragmentPool.runJoinStream(
            parallelJoin.spec,
            { chunks: buildChunks2, columnIndexes: parallelJoin.buildSide.columnIndexes },
            { chunks: probeChunks, columnIndexes: parallelJoin.probeSide.columnIndexes },
            outputTypes
          );
          for await (const chunk of stream) {
            if (chunk.size === 0) continue;
            await currentSink.consume(chunk);
            yield chunk;
            emitted = true;
          }
        } catch (err) {
          if (emitted) throw err;
          yield* emitSerial();
        }
        if (currentSink.finalize) await currentSink.finalize();
      });
    }
  };
}
function buildMergeJoin(node, ctx) {
  const { left, right, buildInput, probeInput, conditionEvaluator } = ctx;
  let mergeBuild = buildInput;
  let mergeProbe = probeInput;
  let mergeBuildKeys = ctx.buildKeys;
  let mergeProbeKeys = ctx.probeKeys;
  let mergeSchema = ctx.resultSchema;
  let mergeMapping = ctx.resultMapping;
  if (node.joinType === JoinType.LEFT && buildInput !== left) {
    mergeBuild = left;
    mergeProbe = right;
    mergeBuildKeys = ctx.probeKeys;
    mergeProbeKeys = ctx.buildKeys;
    mergeSchema = [...left.schema, ...right.schema];
    mergeMapping = combinedMappingOf(left.schema, right.schema);
  } else if (node.joinType === JoinType.RIGHT && buildInput !== right) {
    mergeBuild = right;
    mergeProbe = left;
    mergeBuildKeys = ctx.probeKeys;
    mergeProbeKeys = ctx.buildKeys;
    mergeSchema = [...right.schema, ...left.schema];
    mergeMapping = combinedMappingOf(right.schema, left.schema);
  }
  return {
    schema: mergeSchema,
    columnMapping: mergeMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const leftChunks = registerBufferedChild(graph, currentPipelineId, left);
      const rightChunks = registerBufferedChild(graph, currentPipelineId, right);
      graph.setSource(currentPipelineId, async function* () {
        const mergeJoin = new MergeJoinOperator(
          mergeBuild === left ? leftChunks : rightChunks,
          mergeProbe === left ? leftChunks : rightChunks,
          mergeBuildKeys.map((k) => compileExpression(k, mergeBuild.columnMapping)),
          mergeProbeKeys.map((k) => compileExpression(k, mergeProbe.columnMapping)),
          mergeBuild.schema.length,
          mergeProbe.schema.length,
          node.joinType,
          conditionEvaluator
        );
        const resultChunks = await mergeJoin.execute();
        for (const chunk of resultChunks) {
          await currentSink.consume(chunk);
          yield chunk;
        }
        if (currentSink.finalize) await currentSink.finalize();
      });
    }
  };
}
function buildNestedLoopJoin(node, ctx) {
  const { left, right, buildInput, probeInput, isSemiAnti, isMark, markSchema } = ctx;
  const nlOuter = buildInput === left ? buildInput : probeInput;
  const nlInner = buildInput === left ? probeInput : buildInput;
  const nlMapping = combinedMappingOf(nlOuter.schema, nlInner.schema);
  const nlCondition = node.condition ? compileExpression(node.condition, nlMapping) : null;
  const nlSchema = [...nlOuter.schema, ...nlInner.schema];
  const nlResultMapping = isSemiAnti ? left.columnMapping : nlMapping;
  const nlResultSchema = isSemiAnti ? left.schema : isMark ? markSchema : nlSchema;
  return {
    schema: nlResultSchema,
    columnMapping: nlResultMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const leftChunks = registerBufferedChild(graph, currentPipelineId, left);
      const rightChunks = registerBufferedChild(graph, currentPipelineId, right);
      graph.setSource(currentPipelineId, async function* () {
        const nlJoin = new NestedLoopJoinOperator(
          nlOuter === left ? leftChunks : rightChunks,
          nlInner === left ? leftChunks : rightChunks,
          nlOuter.schema.length,
          nlInner.schema.length,
          node.joinType,
          nlCondition
        );
        const resultChunks = await nlJoin.execute();
        for (const chunk of resultChunks) {
          await currentSink.consume(chunk);
          yield chunk;
        }
        if (currentSink.finalize) await currentSink.finalize();
      });
    }
  };
}
function prepareParallelJoin(executor, node, buildInput, probeInput, buildNode, probeNode, buildKeys, probeKeys, residualCondition, combinedMapping) {
  if (!executor.fragmentPool) return null;
  if (buildKeys.length === 0 || node.joinType === JoinType.CROSS) return null;
  if (executor._estimatePlanRows(node) < Config.parallelJoinThreshold) return null;
  const buildSide = prepareJoinSide(executor, buildNode, buildInput);
  const probeSide = prepareJoinSide(executor, probeNode, probeInput);
  const buildPreserved = node.joinType === JoinType.FULL || node.joinType === JoinType.LEFT && buildNode === node.children[0];
  const spec = buildJoinSpec({
    build: buildSide.spec,
    probe: probeSide.spec,
    buildKeys,
    probeKeys,
    residualCondition,
    joinType: node.joinType,
    buildPreserved,
    uniqueKeys: !!node._dedupeBuild && !residualCondition,
    buildMapping: buildInput.columnMapping,
    probeMapping: probeInput.columnMapping,
    combinedMapping
  });
  if (!spec) return null;
  return { spec, buildSide, probeSide };
}
function prepareJoinSide(executor, planNode, input) {
  const sideSchema = plainSchemaOf(input.schema);
  const buffered = {
    spec: { schema: sideSchema, baseSchema: sideSchema, stages: [] },
    storage: null,
    columnIndexes: null
  };
  const fragment = planNode ? extractStageChain(planNode) : null;
  if (!fragment) return buffered;
  const storage = executor.catalog.getTableStorage(fragment.table);
  if (!storage || typeof storage.scan !== "function") return buffered;
  const storageSchema = storage.getSchema();
  const projected = executor.resolveProjectedColumnIndexes(storageSchema, fragment.scanColumns);
  const columnIndexes = projected || storageSchema.map((_, i) => i);
  const baseSchema = columnIndexes.map((i) => ({
    name: storageSchema[i].name,
    dataType: storageSchema[i].dataType,
    tableAlias: fragment.alias
  }));
  if (!schemasEqual(stagedSchemaOf(baseSchema, fragment.stages), sideSchema)) return buffered;
  return {
    spec: { schema: sideSchema, baseSchema, stages: fragment.stages },
    storage,
    columnIndexes
  };
}
async function runBufferedSerialJoin(makeBuildSide, makeProbeOp, buildChunks2, probeChunks, node, probeColCount) {
  const buildSide = makeBuildSide();
  for (const chunk of buildChunks2) await buildSide.consume(chunk);
  await buildSide.finalize();
  const probeOp = makeProbeOp(buildSide);
  const out = [];
  for (const chunk of probeChunks) {
    const result = await probeOp.process(chunk);
    if (result && result.size > 0) out.push(result);
  }
  if (node.joinType === JoinType.LEFT || node.joinType === JoinType.FULL) {
    const unmatchedRows = buildSide.emitUnmatched(probeColCount);
    if (unmatchedRows.length > 0) out.push(probeOp.buildOutputChunk(unmatchedRows));
  }
  if (probeOp.finalize) {
    await probeOp.finalize({ consume: async (chunk) => {
      out.push(chunk);
    } });
  }
  return out;
}

// src/execution/builders/aggregate-builder.js
init_buffer_shim();
init_logical_plan();

// src/execution/operators/stream-aggregate.js
init_buffer_shim();
init_column();
init_chunk();
init_data_type();
init_dispatch();
var StreamAggregateOperator = class {
  constructor(groupByExtractors, groupByTypes, aggregateDefs) {
    this.groupByExtractors = groupByExtractors;
    this.groupByTypes = groupByTypes;
    this.aggregateDefs = aggregateDefs;
    this.hasCachedValues = aggregateDefs.some((def) => def.valueKey);
  }
  async init() {
  }
  _resolveWasmAggKernel(def) {
    const name = def.name?.toUpperCase();
    if (!name) return null;
    if (name === "SUM" && def.resultType === "FLOAT64") {
      if (globalDispatch.has("sumF64", "FLOAT64")) return { kernelKey: "sumF64", dataType: "FLOAT64", kind: "SUM" };
      if (globalDispatch.has("sumI32", "INT32")) return { kernelKey: "sumI32", dataType: "INT32", kind: "SUM" };
    }
    if (name === "MIN") {
      if (def.resultType === "FLOAT64" && globalDispatch.has("minF64", "FLOAT64")) return { kernelKey: "minF64", dataType: "FLOAT64", kind: "MIN" };
      if (def.resultType === "INT32" && globalDispatch.has("minI32", "INT32")) return { kernelKey: "minI32", dataType: "INT32", kind: "MIN" };
    }
    if (name === "MAX") {
      if (def.resultType === "FLOAT64" && globalDispatch.has("maxF64", "FLOAT64")) return { kernelKey: "maxF64", dataType: "FLOAT64", kind: "MAX" };
      if (def.resultType === "INT32" && globalDispatch.has("maxI32", "INT32")) return { kernelKey: "maxI32", dataType: "INT32", kind: "MAX" };
    }
    if (name === "COUNT") {
      return { kernelKey: "countBits", dataType: "UINT8", kind: "COUNT" };
    }
    if (name === "COUNT_STAR") {
      return { kernelKey: null, dataType: null, kind: "COUNT_STAR" };
    }
    if (name === "AVG" && def.resultType === "FLOAT64") {
      if (globalDispatch.has("sumF64", "FLOAT64")) return { kernelKey: "sumF64", dataType: "FLOAT64", kind: "AVG" };
    }
    return null;
  }
  async _tryWasmUngrouped(chunks) {
    if (this.groupByExtractors.length !== 0) return null;
    if (!globalDispatch || globalDispatch.kernels.size === 0) return null;
    const accumulators = this.aggregateDefs.map((def) => def.createAccumulator());
    for (const chunk of chunks) {
      if (chunk.selectionVector || chunk.size === 0) return null;
      for (let a = 0; a < this.aggregateDefs.length; a++) {
        const def = this.aggregateDefs[a];
        const resolved = this._resolveWasmAggKernel(def);
        if (!resolved) return null;
        const acc = accumulators[a];
        if (resolved.kind === "COUNT_STAR") {
          acc.count += chunk.size;
          continue;
        }
        if (resolved.kind === "COUNT") {
          if (!def._wasmColIndex && def._wasmColIndex !== 0) return null;
          const column = chunk.columns[def._wasmColIndex];
          if (!column) return null;
          if (!column.hasNulls) {
            acc.count += chunk.size;
          } else {
            const kernel2 = globalDispatch.lookup("countBits", "UINT8");
            if (!kernel2) return null;
            const nonNullCount = await kernel2(column.nullBitmap, chunk.size);
            acc.count += nonNullCount;
          }
          continue;
        }
        if (resolved.kind === "AVG") {
          if (!def._wasmColIndex && def._wasmColIndex !== 0) return null;
          const column = chunk.columns[def._wasmColIndex];
          if (!column || !column.data) return null;
          if (column.dataType !== "FLOAT64") return null;
          const rawData2 = column.data.subarray(0, chunk.size);
          const kernel2 = globalDispatch.lookup(resolved.kernelKey, resolved.dataType);
          if (!kernel2) return null;
          const sum2 = await kernel2(rawData2);
          let nonNullCount;
          if (!column.hasNulls) {
            nonNullCount = chunk.size;
          } else {
            const countKernel = globalDispatch.lookup("countBits", "UINT8");
            if (!countKernel) return null;
            nonNullCount = await countKernel(column.nullBitmap, chunk.size);
          }
          acc.sum += sum2;
          acc.count += nonNullCount;
          continue;
        }
        let rawData = null;
        if (def._wasmColIndex !== void 0 && def._wasmColIndex !== null) {
          const column = chunk.columns[def._wasmColIndex];
          if (column && column.data) {
            const colType = column.dataType;
            if (resolved.dataType === "FLOAT64" && colType === "FLOAT64") {
              rawData = column.data.subarray(0, chunk.size);
            } else if (resolved.dataType === "INT32" && (colType === "INT32" || colType === "DATE")) {
              rawData = column.data.subarray(0, chunk.size);
            }
          }
        }
        if (!rawData && def._sourceExpr && isVectorizableExpr(def._sourceExpr)) {
          const vectorResult = await evalVectorized(def._sourceExpr, chunk, def._columnMapping, chunk.size);
          if (vectorResult instanceof Float64Array) rawData = vectorResult;
        }
        if (!rawData) return null;
        const kernel = globalDispatch.lookup(resolved.kernelKey, resolved.dataType);
        if (!kernel) return null;
        acc.add(await kernel(rawData));
      }
    }
    const row = accumulators.map((a) => a.result());
    const columns = [];
    for (let a = 0; a < this.aggregateDefs.length; a++) {
      const col2 = new Column(this.aggregateDefs[a].resultType, 1);
      col2.set(0, typeof row[a] === "bigint" ? Number(row[a]) : row[a]);
      col2.length = 1;
      columns.push(col2);
    }
    return [new DataChunk(columns, 1)];
  }
  async execute(chunks) {
    const wasmResult = await this._tryWasmUngrouped(chunks);
    if (wasmResult !== null) return wasmResult;
    const outputRows = [];
    let currentKey = null;
    let groupValues = null;
    let accumulators = null;
    for (const chunk of chunks) {
      for (let i = 0; i < chunk.size; i++) {
        const rowIdx = chunk.activeRowIndex(i);
        const key = this.extractGroupKey(chunk, rowIdx);
        if (currentKey !== key) {
          if (accumulators !== null) {
            const row = [...groupValues];
            for (let a = 0; a < accumulators.length; a++) {
              row.push(accumulators[a].result());
            }
            outputRows.push(row);
          }
          currentKey = key;
          groupValues = this.groupByExtractors.map((fn) => fn(chunk, rowIdx));
          accumulators = this.aggregateDefs.map((def) => def.createAccumulator());
        }
        if (accumulators !== null) {
          const valueCache = this.hasCachedValues ? /* @__PURE__ */ Object.create(null) : null;
          for (let a = 0; a < this.aggregateDefs.length; a++) {
            const def = this.aggregateDefs[a];
            let val;
            if (valueCache && def.valueKey) {
              if (Object.prototype.hasOwnProperty.call(valueCache, def.valueKey)) {
                val = valueCache[def.valueKey];
              } else {
                val = def.extractValue(chunk, rowIdx);
                valueCache[def.valueKey] = val;
              }
            } else {
              val = def.extractValue(chunk, rowIdx);
            }
            accumulators[a].add(val);
          }
        }
      }
    }
    if (accumulators !== null) {
      const row = [...groupValues];
      for (let a = 0; a < accumulators.length; a++) {
        row.push(accumulators[a].result());
      }
      outputRows.push(row);
    } else if (this.groupByExtractors.length === 0) {
      const acc = this.aggregateDefs.map((def) => def.createAccumulator());
      outputRows.push(acc.map((a) => a.result()));
    }
    if (outputRows.length === 0) return [];
    const totalCols = this.groupByExtractors.length + this.aggregateDefs.length;
    const columns = [];
    for (let g = 0; g < this.groupByExtractors.length; g++) {
      columns.push(new Column(this.groupByTypes[g] || DataType.VARCHAR, outputRows.length));
    }
    for (let a = 0; a < this.aggregateDefs.length; a++) {
      columns.push(new Column(this.aggregateDefs[a].resultType, outputRows.length));
    }
    for (let r = 0; r < outputRows.length; r++) {
      const row = outputRows[r];
      for (let c = 0; c < totalCols; c++) {
        columns[c].set(r, typeof row[c] === "bigint" ? Number(row[c]) : row[c]);
      }
    }
    for (const col2 of columns) col2.length = outputRows.length;
    return [new DataChunk(columns, outputRows.length)];
  }
  extractGroupKey(chunk, rowIdx) {
    if (this.groupByExtractors.length === 0) return "__ALL__";
    return this.groupByExtractors.map((fn) => {
      const v = fn(chunk, rowIdx);
      return typeof v === "bigint" ? v.toString() : String(v);
    }).join("|");
  }
};

// src/execution/builders/aggregate-builder.js
init_config();
async function buildAggregate(executor, node) {
  const child = await executor.buildPipeline(node.children[0]);
  const groupByEvals = (node.groupBy || []).map(
    (expr2) => compileExpression(expr2, child.columnMapping)
  );
  const groupByTypes = (node.groupBy || []).map(
    (expr2) => executor.normalizeExecType(expr2?.dataType || expr2?.resultType || "VARCHAR")
  );
  const aggDefs = buildAggregateDefs(node.aggregates, child.columnMapping);
  const schema = [
    ...(node.groupBy || []).map((expr2, i) => ({
      name: expr2?.columnName || `group${i}`,
      dataType: groupByTypes[i],
      tableAlias: expr2?.tableAlias || ""
    })),
    ...node.aggregates.map((agg, i) => ({
      name: agg.outputName || agg.name.toLowerCase(),
      dataType: executor.normalizeAggResultType(agg),
      tableAlias: ""
    }))
  ];
  const columnMapping = aggregateSchemaMapping(schema, node.groupBy || [], node.aggregates);
  if (node.physicalStrategy === PhysicalStrategy.STREAM) {
    return {
      schema,
      columnMapping,
      register: (graph, currentPipelineId, currentSink) => {
        const childChunks = registerBufferedChild(graph, currentPipelineId, child);
        graph.setSource(currentPipelineId, async function* () {
          const aggOp = new StreamAggregateOperator(groupByEvals, groupByTypes, aggDefs);
          const resultChunks = await aggOp.execute(childChunks);
          for (const chunk of resultChunks) {
            await currentSink.consume(chunk);
            yield chunk;
          }
          if (currentSink.finalize) await currentSink.finalize();
        });
      }
    };
  }
  const serialCompiled = {
    schema,
    columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const aggOp = new HashAggregateOperator(groupByEvals, groupByTypes, aggDefs);
      const aggSink = {
        async consume(chunk) {
          await aggOp.consume(chunk);
        },
        async finalize() {
        }
      };
      const childPipelineId = graph.createPipeline(aggSink);
      child.register(graph, childPipelineId, aggSink);
      graph.addDependency(currentPipelineId, childPipelineId);
      graph.setSource(currentPipelineId, async function* () {
        const resultChunks = await aggOp.finalize();
        for (const chunk of resultChunks) {
          await currentSink.consume(chunk);
          yield chunk;
        }
        if (currentSink.finalize) await currentSink.finalize();
      });
    }
  };
  const parallel = prepareParallelAggregate(executor, node);
  if (!parallel) return serialCompiled;
  return {
    schema,
    columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const rowCount = parallel.storage.rowCount();
      const withinMemory = rowCount * parallel.estimatedRowBytes <= Config.parallelAggMemoryBytes;
      if (rowCount < Config.parallelAggThreshold || !withinMemory) {
        serialCompiled.register(graph, currentPipelineId, currentSink);
        return;
      }
      graph.setSource(currentPipelineId, async function* () {
        let resultChunks = null;
        try {
          const chunks = [];
          for await (const chunk of parallel.storage.scan()) chunks.push(chunk);
          resultChunks = await executor.fragmentPool.runAggregate(parallel.spec, parallel.columnIndexes, chunks, {
            spillDir: executor.tempManager.allocate("spill", "pagg")
          });
        } catch (_) {
          resultChunks = null;
        }
        if (resultChunks === null) {
          resultChunks = await executor._executeSubPipeline(serialCompiled);
        }
        for (const chunk of resultChunks) {
          await currentSink.consume(chunk);
          yield chunk;
        }
        if (currentSink.finalize) await currentSink.finalize();
      });
    }
  };
}
function prepareParallelAggregate(executor, node) {
  if (!executor.fragmentPool) return null;
  const fragment = extractAggregateFragment(node);
  if (!fragment) return null;
  const storage = executor.catalog.getTableStorage(fragment.table);
  if (!storage || typeof storage.scan !== "function") return null;
  const built = buildFragmentSpec(fragment, node, storage.getSchema());
  if (!built) return null;
  return { storage, ...built };
}
function aggregateSchemaMapping(schema, groupBy, aggregates) {
  const columnMapping = /* @__PURE__ */ new Map();
  let idx = 0;
  for (const col2 of schema) {
    const key = col2.tableAlias ? `${col2.tableAlias}.${col2.name}`.toUpperCase() : col2.name.toUpperCase();
    columnMapping.set(key, idx);
    columnMapping.set(col2.name.toUpperCase(), idx);
    idx++;
  }
  const groupByCount = groupBy.length;
  for (let a = 0; a < aggregates.length; a++) {
    columnMapping.set(aggExprKey(aggregates[a]), groupByCount + a);
  }
  return columnMapping;
}
async function buildPartialAggregate(executor, node) {
  const child = await executor.buildPipeline(node.children[0]);
  const groupByEvals = (node.groupBy || []).map(
    (expr2) => compileExpression(expr2, child.columnMapping)
  );
  const groupByTypes = (node.groupBy || []).map(
    (expr2) => executor.normalizeExecType(expr2?.dataType || expr2?.resultType || "VARCHAR")
  );
  const aggDefs = [];
  const aggSchemaCols = [];
  const mappingAggs = [];
  for (const agg of node.aggregates) {
    const funcName = (agg.func || agg.name || "").toUpperCase();
    const valueExtractor = agg.args && agg.args.length > 0 ? compileExpression(agg.args[0], child.columnMapping) : () => 1;
    const extract = (chunk, rowIdx) => {
      const val = valueExtractor(chunk, rowIdx);
      return typeof val === "bigint" ? Number(val) : val;
    };
    if (funcName === "AVG_PARTIAL") {
      aggDefs.push({ name: "SUM", resultType: "FLOAT64", createAccumulator: getAccumulatorFactory("SUM"), extractValue: extract });
      aggDefs.push({ name: "COUNT", resultType: "FLOAT64", createAccumulator: getAccumulatorFactory("COUNT"), extractValue: extract });
      aggSchemaCols.push({ name: "_avg_sum", dataType: "FLOAT64", tableAlias: "" });
      aggSchemaCols.push({ name: "_avg_count", dataType: "FLOAT64", tableAlias: "" });
      mappingAggs.push({ func: "SUM", args: agg.args }, { func: "COUNT", args: agg.args });
      continue;
    }
    aggDefs.push({
      name: agg.func || agg.name,
      resultType: executor.normalizeAggResultType(agg),
      createAccumulator: getAccumulatorFactory(agg.func || agg.name, agg.distinct),
      extractValue: extract
    });
    aggSchemaCols.push({ name: (agg.func || agg.name || "").toLowerCase(), dataType: executor.normalizeAggResultType(agg), tableAlias: "" });
    mappingAggs.push(agg);
  }
  const schema = [
    ...(node.groupBy || []).map((expr2, i) => ({
      name: expr2?.columnName || `group${i}`,
      dataType: groupByTypes[i],
      tableAlias: expr2?.tableAlias || ""
    })),
    ...aggSchemaCols
  ];
  const columnMapping = aggregateSchemaMapping(schema, node.groupBy || [], mappingAggs);
  return {
    schema,
    columnMapping,
    register: registerHashAggregate(child, () => new HashAggregateOperator(groupByEvals, groupByTypes, aggDefs))
  };
}
async function buildFinalAggregate(executor, node) {
  const child = await executor.buildPipeline(node.children[0]);
  const groupByCount = (node.groupBy || []).length;
  const groupByEvals = (node.groupBy || []).map((_expr, i) => {
    const colIdx = i;
    return (chunk, rowIdx) => chunk.columns[colIdx]?.get(rowIdx);
  });
  const groupByTypes = (node.groupBy || []).map(
    (expr2) => executor.normalizeExecType(expr2?.dataType || expr2?.resultType || "VARCHAR")
  );
  const finalAggs = node.aggregates;
  const partialAggs = node.partialAggregates || finalAggs;
  const partialWidth = (agg) => (agg.func || agg.name || "").toUpperCase() === "AVG_PARTIAL" ? 2 : 1;
  const partialStarts = [];
  let partialOffset = groupByCount;
  for (let i = 0; i < finalAggs.length; i++) {
    partialStarts.push(partialOffset);
    partialOffset += partialWidth(partialAggs[i] || finalAggs[i]);
  }
  const aggDefs = finalAggs.map((agg, aggIdx) => {
    const funcName = (agg.func || agg.name || "").toUpperCase();
    const start = partialStarts[aggIdx];
    if (funcName === "AVG_FINAL") {
      return {
        name: "AVG",
        resultType: executor.normalizeAggResultType(agg),
        createAccumulator: getAccumulatorFactory("AVG_FINAL", false),
        extractValue: (chunk, rowIdx) => {
          const s = chunk.columns[start]?.get(rowIdx);
          const c = chunk.columns[start + 1]?.get(rowIdx);
          return [
            typeof s === "bigint" ? Number(s) : s,
            typeof c === "bigint" ? Number(c) : c
          ];
        }
      };
    }
    return {
      name: funcName,
      resultType: executor.normalizeAggResultType(agg),
      createAccumulator: getAccumulatorFactory(funcName, false),
      extractValue: (chunk, rowIdx) => {
        const val = chunk.columns[start]?.get(rowIdx);
        return typeof val === "bigint" ? Number(val) : val;
      }
    };
  });
  const schema = [
    ...(node.groupBy || []).map((expr2, i) => ({
      name: expr2?.columnName || `group${i}`,
      dataType: groupByTypes[i],
      tableAlias: expr2?.tableAlias || ""
    })),
    ...finalAggs.map((agg) => ({
      name: (agg.name || agg.func || "").toLowerCase(),
      dataType: executor.normalizeAggResultType(agg),
      tableAlias: ""
    }))
  ];
  const columnMapping = aggregateSchemaMapping(schema, node.groupBy || [], finalAggs);
  return {
    schema,
    columnMapping,
    register: registerHashAggregate(child, () => new HashAggregateOperator(groupByEvals, groupByTypes, aggDefs))
  };
}
function registerHashAggregate(child, makeAggOp) {
  return (graph, currentPipelineId, currentSink) => {
    const aggOp = makeAggOp();
    const aggSink = {
      async consume(chunk) {
        await aggOp.consume(chunk);
      },
      async finalize() {
      }
    };
    const childPipelineId = graph.createPipeline(aggSink);
    child.register(graph, childPipelineId, aggSink);
    graph.addDependency(currentPipelineId, childPipelineId);
    graph.setSource(currentPipelineId, async function* () {
      const resultChunks = await aggOp.finalize();
      for (const chunk of resultChunks) {
        await currentSink.consume(chunk);
        yield chunk;
      }
      if (currentSink.finalize) await currentSink.finalize();
    });
  };
}

// src/execution/builders/cte-builders.js
init_buffer_shim();

// src/execution/operators/dependent-join.js
init_buffer_shim();
init_column();
init_chunk();
var DependentJoinOperator = class {
  constructor(subqueryType, outerSchema) {
    this.subqueryType = subqueryType;
    this.outerSchema = outerSchema;
    this.resultRows = [];
    this.resultSchema = this.subqueryType === "SCALAR" ? [...outerSchema, { name: "_scalar", dataType: "FLOAT64", tableAlias: "" }] : outerSchema;
  }
  async processOuterRow(outerRow, subResultChunks) {
    const subRows = [];
    for (const chunk of subResultChunks) {
      for (let i = 0; i < chunk.size; i++) {
        const row = [];
        for (let c = 0; c < chunk.columns.length; c++) {
          row.push(chunk.columns[c].get(chunk.activeRowIndex(i)));
        }
        subRows.push(row);
      }
    }
    if (this.subqueryType === "EXISTS") {
      if (subRows.length > 0) this.resultRows.push(outerRow);
    } else if (this.subqueryType === "NOT_EXISTS") {
      if (subRows.length === 0) this.resultRows.push(outerRow);
    } else if (this.subqueryType === "SCALAR") {
      const scalarVal = subRows.length > 0 ? subRows[0][0] : null;
      this.resultRows.push([...outerRow, scalarVal]);
    } else if (this.subqueryType === "IN") {
      if (subRows.length > 0) this.resultRows.push(outerRow);
    } else if (this.subqueryType === "NOT_IN") {
      if (subRows.length === 0) this.resultRows.push(outerRow);
    } else {
      this.resultRows.push(outerRow);
    }
  }
  async finalize() {
    if (this.resultRows.length === 0) {
      return [];
    }
    const colCount = this.resultSchema.length;
    const cols = [];
    for (let c = 0; c < colCount; c++) {
      const col2 = new Column(this.resultSchema[c].dataType || "VARCHAR", this.resultRows.length);
      for (let r = 0; r < this.resultRows.length; r++) {
        col2.set(r, this.resultRows[r][c]);
      }
      col2.length = this.resultRows.length;
      cols.push(col2);
    }
    return [new DataChunk(cols, this.resultRows.length)];
  }
};

// src/execution/builders/cte-builders.js
init_column();
init_chunk();
init_config();
async function buildCTEAnchor(executor, node) {
  const producer = await executor.buildPipeline(node.children[0]);
  executor.cteDefinitions.set(node.cteName.toUpperCase(), node.children[0]);
  const consumer = await executor.buildPipeline(node.children[1]);
  return {
    schema: consumer.schema,
    columnMapping: consumer.columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      const cteChunks = [];
      const cteSink = { consume: async (c) => cteChunks.push(c) };
      const producerPipelineId = graph.createPipeline(cteSink);
      producer.register(graph, producerPipelineId, cteSink);
      cteSink.finalize = async () => {
        executor.cteResults.set(node.cteName.toUpperCase(), { chunks: cteChunks, schema: producer.schema, columnMapping: producer.columnMapping });
      };
      graph.addDependency(currentPipelineId, producerPipelineId);
      consumer.register(graph, currentPipelineId, currentSink);
    }
  };
}
async function buildCTEScan(executor, node) {
  const ctePlan = executor.findCTEPlan(node.cteName);
  if (!ctePlan) throw new Error(`CTE not found: ${node.cteName}`);
  const compiledCTE = await executor.buildPipeline(ctePlan);
  return {
    schema: compiledCTE.schema,
    columnMapping: compiledCTE.columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      graph.setSource(currentPipelineId, async function* () {
        let stored = executor.cteResults.get(node.cteName.toUpperCase());
        if (!stored) {
          const cteChunks = [];
          const cteSink = {
            async consume(c) {
              cteChunks.push(c);
            },
            async finalize() {
            }
          };
          const cteGraph = new PipelineGraph();
          const ctePipelineId = cteGraph.createPipeline(cteSink);
          compiledCTE.register(cteGraph, ctePipelineId, cteSink);
          const scheduler = new TaskScheduler();
          await scheduler.schedule(cteGraph);
          stored = {
            chunks: cteChunks,
            schema: compiledCTE.schema,
            columnMapping: compiledCTE.columnMapping
          };
          executor.cteResults.set(node.cteName.toUpperCase(), stored);
        }
        const clonedChunks = stored.chunks.map((chunk) => {
          const cols = chunk.columns.map((col2) => {
            const newCol = new Column(col2.dataType, chunk.size);
            for (let i = 0; i < chunk.size; i++) {
              newCol.set(i, col2.get(chunk.activeRowIndex(i)));
            }
            newCol.length = chunk.size;
            return newCol;
          });
          return new DataChunk(cols, chunk.size);
        });
        for (const chunk of clonedChunks) {
          await currentSink.consume(chunk);
          yield chunk;
        }
      });
    }
  };
}
async function buildMaterialize(executor, node) {
  const child = await executor.buildPipeline(node.children[0]);
  return {
    schema: child.schema,
    columnMapping: child.columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      child.register(graph, currentPipelineId, currentSink);
    }
  };
}
async function buildDependentJoin(executor, node) {
  const outer = await executor.buildPipeline(node.children[0]);
  const dummyOp = new DependentJoinOperator(node.subqueryType, outer.schema);
  return {
    schema: dummyOp.resultSchema,
    columnMapping: executor.buildSchemaMapping(dummyOp.resultSchema, ""),
    register: (graph, currentPipelineId, currentSink) => {
      const outerChunks = [];
      const outerSink = { consume: async (c) => outerChunks.push(c) };
      const outerPipelineId = graph.createPipeline(outerSink);
      outer.register(graph, outerPipelineId, outerSink);
      graph.addDependency(currentPipelineId, outerPipelineId);
      graph.setSource(currentPipelineId, async function* () {
        const runtimeOp = new DependentJoinOperator(node.subqueryType, outer.schema);
        const isCorrelated = (node.correlatedColumns || []).length > 0;
        let cachedInnerChunks = null;
        for (const outerChunk of outerChunks) {
          const outerRows = outerChunk.toRows();
          for (const outerRow of outerRows) {
            if (!isCorrelated && cachedInnerChunks !== null) {
              await runtimeOp.processOuterRow(outerRow, cachedInnerChunks);
              continue;
            }
            const innerPipeline = await executor.buildPipeline(node.children[1]);
            const innerChunks = [];
            const innerGraph = new PipelineGraph();
            const innerSink = { consume: async (c) => innerChunks.push(c) };
            const innerPipelineId = innerGraph.createPipeline(innerSink);
            innerPipeline.register(innerGraph, innerPipelineId, innerSink);
            await new TaskScheduler(Config.dependentJoinConcurrency).schedule(innerGraph);
            if (!isCorrelated) cachedInnerChunks = innerChunks;
            await runtimeOp.processOuterRow(outerRow, innerChunks);
          }
        }
        const resultChunks = await runtimeOp.finalize();
        for (const chunk of resultChunks) {
          await currentSink.consume(chunk);
          yield chunk;
        }
      });
    }
  };
}

// src/execution/builders/exchange-builders.js
init_buffer_shim();
function buildKeyExtractors(partitionKeys, columnMapping) {
  if (!partitionKeys || partitionKeys.length === 0) return [];
  return partitionKeys.map((key) => {
    const evalFn = compileExpression(key, columnMapping);
    return (chunk, rowIdx) => evalFn(chunk, rowIdx);
  });
}
async function buildExchange(executor, node) {
  const child = await executor.buildPipeline(node.children[0]);
  if (executor._distributedContext) {
    const { transport, sourceNodes, channelId, exchangeType } = executor._distributedContext.getExchangeConfig(node);
    const isReceiver = executor._distributedContext.role === "receiver";
    if (isReceiver) {
      const { ExchangeReceiver } = await import("../../distributed/execution/exchange-operator.js");
      const receiver = new ExchangeReceiver(transport, sourceNodes, { channelId });
      await receiver.init();
      return {
        schema: child.schema,
        columnMapping: child.columnMapping,
        register: (graph, currentPipelineId, currentSink) => {
          graph.setSource(currentPipelineId, async function* () {
            for await (const chunk of receiver.generate()) {
              await currentSink.consume(chunk);
              yield chunk;
            }
            receiver.cleanup();
            if (currentSink.finalize) await currentSink.finalize();
          });
        }
      };
    }
    const { ExchangeSender } = await import("../../distributed/execution/exchange-operator.js");
    const sender = new ExchangeSender(transport, node._targetNodes || [], {
      exchangeType: exchangeType || node.exchangeType,
      channelId,
      partitionCount: node.partitionCount,
      keyExtractors: buildKeyExtractors(node.partitionKeys, child.columnMapping)
    });
    return {
      schema: child.schema,
      columnMapping: child.columnMapping,
      register: (graph, currentPipelineId, currentSink) => {
        const childSink = {
          get cancelToken() {
            return currentSink.cancelToken;
          },
          async consume(chunk) {
            await sender.consume(chunk);
            await currentSink.consume(chunk);
          },
          async finalize() {
            await sender.finalize();
            if (currentSink.finalize) await currentSink.finalize();
          }
        };
        child.register(graph, currentPipelineId, childSink);
      }
    };
  }
  return {
    schema: child.schema,
    columnMapping: child.columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      child.register(graph, currentPipelineId, currentSink);
    }
  };
}
async function buildMergeExchange(executor, node) {
  const child = await executor.buildPipeline(node.children[0]);
  if (executor._distributedContext) {
    const { transport, sourceNodes, channelId } = executor._distributedContext.getMergeExchangeConfig(node);
    const { MergeExchangeOperator } = await import("../../distributed/execution/merge-exchange.js");
    const merge = new MergeExchangeOperator(transport, sourceNodes, {
      orderKeys: node.orderKeys,
      limit: node.limit,
      channelId
    });
    await merge.init();
    return {
      schema: child.schema,
      columnMapping: child.columnMapping,
      register: (graph, currentPipelineId, currentSink) => {
        graph.setSource(currentPipelineId, async function* () {
          for await (const chunk of merge.generate()) {
            await currentSink.consume(chunk);
            yield chunk;
          }
          merge.cleanup();
          if (currentSink.finalize) await currentSink.finalize();
        });
      }
    };
  }
  return {
    schema: child.schema,
    columnMapping: child.columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      child.register(graph, currentPipelineId, currentSink);
    }
  };
}
async function buildExchangeReceive(executor, node) {
  const receivers = executor._exchangeReceivers;
  const fragmentIds = node.sourceFragmentIds || [];
  const matchingReceivers = [];
  if (receivers) {
    for (const fid of fragmentIds) {
      const receiver = receivers.get(fid);
      if (receiver) matchingReceivers.push(receiver);
    }
  }
  const schema = node.schema || [];
  const columnMapping = /* @__PURE__ */ new Map();
  schema.forEach((col2, idx) => {
    if (col2.tableAlias) columnMapping.set(`${col2.tableAlias}.${col2.name}`.toUpperCase(), idx);
    columnMapping.set(col2.name.toUpperCase(), idx);
  });
  return {
    schema,
    columnMapping,
    register: (graph, currentPipelineId, currentSink) => {
      graph.setSource(currentPipelineId, async function* () {
        for (const receiver of matchingReceivers) {
          for await (const chunk of receiver.generate()) {
            if (chunk && chunk.size > 0) {
              await currentSink.consume(chunk);
              yield chunk;
            }
          }
          receiver.cleanup();
        }
        if (currentSink.finalize) await currentSink.finalize();
      });
    }
  };
}

// src/execution/query-executor.js
var BUILDERS = {
  [PlanNodeType.SCAN]: buildScan,
  [PlanNodeType.INDEX_SCAN]: buildIndexScan,
  [PlanNodeType.FILTER]: buildFilter,
  [PlanNodeType.PROJECT]: buildProject,
  [PlanNodeType.JOIN]: buildJoin,
  [PlanNodeType.AGGREGATE]: buildAggregate,
  [PlanNodeType.SORT]: buildSort,
  [PlanNodeType.LIMIT]: buildLimit,
  [PlanNodeType.DISTINCT]: buildDistinct,
  [PlanNodeType.UNION]: buildUnion,
  [PlanNodeType.CTE_ANCHOR]: buildCTEAnchor,
  [PlanNodeType.CTE_SCAN]: buildCTEScan,
  [PlanNodeType.MATERIALIZE]: buildMaterialize,
  [PlanNodeType.DEPENDENT_JOIN]: buildDependentJoin,
  [PlanNodeType.TOP_N]: buildTopN,
  [PlanNodeType.WINDOW]: buildWindow,
  [PlanNodeType.EMPTY]: buildEmpty,
  [PlanNodeType.SINGLE_ROW]: buildSingleRow,
  [PlanNodeType.EXCHANGE]: buildExchange,
  [PlanNodeType.PARTIAL_AGGREGATE]: buildPartialAggregate,
  [PlanNodeType.FINAL_AGGREGATE]: buildFinalAggregate,
  [PlanNodeType.MERGE_EXCHANGE]: buildMergeExchange,
  [PlanNodeType.EXCHANGE_RECEIVE]: buildExchangeReceive
};
var QueryExecutor = class {
  constructor(catalog, tempManager, storageBackend = null) {
    this.catalog = catalog;
    this.tempManager = tempManager;
    this.storageBackend = storageBackend ?? new MemoryStorageBackend();
    this.cteResults = /* @__PURE__ */ new Map();
    this.cteDefinitions = /* @__PURE__ */ new Map();
    this.workerPool = null;
    this.parallelDispatch = null;
    this.fragmentPool = null;
  }
  setParallelContext(workerPool, parallelDispatch, fragmentPool = null) {
    this.workerPool = workerPool;
    this.parallelDispatch = parallelDispatch;
    this.fragmentPool = fragmentPool;
  }
  setDistributedContext(ctx) {
    this._distributedContext = ctx;
  }
  _shouldParallelize(storage) {
    return this.workerPool && this.parallelDispatch && storage.rowCount() >= Config.parallelThreshold;
  }
  async execute(logicalPlan, outputColumns, streaming = false) {
    const sink = await this.executePlan(logicalPlan, streaming);
    const columnNames = outputColumns.map((c) => c.name);
    return { sink, columnNames };
  }
  async executeStreaming(logicalPlan, outputColumns) {
    return this.execute(logicalPlan, outputColumns, true);
  }
  async executePlan(logicalPlan, streaming = false) {
    this.cteResults.clear();
    const graph = new PipelineGraph();
    const resultSink = new ResultSink(streaming);
    await resultSink.init();
    const rootPipelineId = graph.createPipeline(resultSink);
    const compiledRoot = await this.buildPipeline(logicalPlan);
    compiledRoot.register(graph, rootPipelineId, resultSink);
    const scheduler = new TaskScheduler();
    if (streaming) {
      const pipelinePromise = scheduler.schedule(graph);
      pipelinePromise.catch((err) => resultSink.error(err));
      return resultSink;
    }
    await scheduler.schedule(graph);
    return resultSink;
  }
  async buildPipeline(node) {
    const builder = BUILDERS[node.type];
    if (!builder) {
      throw new Error(`Unsupported plan node: ${node.type}`);
    }
    return builder(this, node);
  }
  resolveProjectedColumnIndexes(storageSchema, planColumns) {
    if (!planColumns || planColumns.length === 0 || planColumns.length >= storageSchema.length) {
      return null;
    }
    const indexes = [];
    for (const col2 of planColumns) {
      const idx = storageSchema.findIndex((s) => s.name.toUpperCase() === col2.name.toUpperCase());
      if (idx < 0) return null;
      indexes.push(idx);
    }
    return indexes;
  }
  buildSchemaMapping(schema, alias) {
    const mapping = /* @__PURE__ */ new Map();
    for (let i = 0; i < schema.length; i++) {
      const col2 = schema[i];
      const tableAlias = col2.tableAlias || alias || "";
      const key = `${tableAlias}.${col2.name}`.toUpperCase();
      mapping.set(key, i);
      if (!mapping.has(col2.name.toUpperCase())) {
        mapping.set(col2.name.toUpperCase(), i);
      }
    }
    return mapping;
  }
  findCTEPlan(cteName) {
    const key = cteName.toUpperCase();
    return this.cteDefinitions?.get(key) || null;
  }
  _estimatePlanRows(planNode) {
    let total = 0;
    const stack = [planNode];
    while (stack.length > 0) {
      const current = stack.pop();
      if (!current) continue;
      if (current.type === PlanNodeType.SCAN || current.type === PlanNodeType.INDEX_SCAN) {
        const storage = this.catalog.getTableStorage(current.table);
        if (storage) total += storage.rowCount();
      }
      for (const child of current.children || []) stack.push(child);
    }
    return total;
  }
  async _executeSubPipeline(compiled) {
    const chunks = [];
    const sink = {
      consume: async (chunk) => {
        chunks.push(chunk);
      },
      finalize: async () => {
      }
    };
    const graph = new PipelineGraph();
    const pipelineId = graph.createPipeline(sink);
    compiled.register(graph, pipelineId, sink);
    await new TaskScheduler().schedule(graph);
    return chunks;
  }
  _prepareParallelJoin(node, buildInput, probeInput, buildNode, probeNode, buildKeys, probeKeys, residualCondition, combinedMapping) {
    return prepareParallelJoin(this, node, buildInput, probeInput, buildNode, probeNode, buildKeys, probeKeys, residualCondition, combinedMapping);
  }
  _runBufferedSerialJoin(makeBuildSide, makeProbeOp, buildChunks2, probeChunks, node, probeColCount) {
    return runBufferedSerialJoin(makeBuildSide, makeProbeOp, buildChunks2, probeChunks, node, probeColCount);
  }
  normalizeExecType(dt) {
    return normalizeExecType(dt);
  }
  normalizeAggResultType(agg) {
    return normalizeAggResultType(agg);
  }
};

// src/execution/query-result.js
init_buffer_shim();
var QueryResult = class {
  constructor(columnNames, sink) {
    this._columnNames = columnNames;
    this._sink = sink;
  }
  get columns() {
    return this._columnNames;
  }
  async toArray() {
    const result = [];
    for await (const chunk of this._sink) {
      for (let i = 0; i < chunk.size; i++) {
        const rowIdx = chunk.activeRowIndex(i);
        const obj = {};
        for (let j = 0; j < this._columnNames.length; j++) {
          let val = chunk.columns[j].get(rowIdx);
          if (typeof val === "bigint") val = Number(val);
          obj[this._columnNames[j]] = val;
        }
        result.push(obj);
      }
    }
    return result;
  }
  async *[Symbol.asyncIterator]() {
    for await (const chunk of this._sink) {
      for (let i = 0; i < chunk.size; i++) {
        const rowIdx = chunk.activeRowIndex(i);
        const obj = {};
        for (let j = 0; j < this._columnNames.length; j++) {
          let val = chunk.columns[j].get(rowIdx);
          if (typeof val === "bigint") val = Number(val);
          obj[this._columnNames[j]] = val;
        }
        yield obj;
      }
    }
  }
  async *chunks() {
    for await (const chunk of this._sink) {
      yield chunk;
    }
  }
};

// src/catalog/statistics.js
init_buffer_shim();
var HISTOGRAM_BUCKETS = 64;
var MCV_COUNT = 10;
var ColumnStatistics = class {
  constructor({ ndv = 0, min: min2 = null, max: max2 = null, nullFraction = 0, histogram = null, mcv = null, avgWidth = 8, avgLength = null } = {}) {
    this.ndv = ndv;
    this.min = min2;
    this.max = max2;
    this.nullFraction = nullFraction;
    this.histogram = histogram;
    this.mcv = mcv;
    this.avgWidth = avgWidth;
    this.avgLength = avgLength;
  }
};
var TableStatistics = class {
  constructor(rowCount, columnStats = /* @__PURE__ */ new Map(), correlations = /* @__PURE__ */ new Map()) {
    this.rowCount = rowCount;
    this.columnStats = columnStats;
    this.correlations = correlations;
  }
  getColumnStats(columnName) {
    return this.columnStats.get(columnName.toUpperCase()) || null;
  }
  setColumnStats(columnName, stats) {
    this.columnStats.set(columnName.toUpperCase(), stats);
  }
  _correlationKey(colA, colB) {
    const a = colA.toUpperCase();
    const b = colB.toUpperCase();
    return a < b ? `${a}:${b}` : `${b}:${a}`;
  }
  getCorrelation(colA, colB) {
    return this.correlations.get(this._correlationKey(colA, colB)) ?? null;
  }
  setCorrelation(colA, colB, value) {
    this.correlations.set(this._correlationKey(colA, colB), value);
  }
  get avgRowWidth() {
    let width = 0;
    for (const cs of this.columnStats.values()) {
      width += cs.avgWidth || 8;
    }
    return width || 64;
  }
};
var EquiDepthHistogram = class {
  constructor(boundaries, numRows) {
    this.boundaries = boundaries;
    this.numBuckets = boundaries.length;
    this.numRows = numRows;
    this.rowsPerBucket = numRows / Math.max(this.numBuckets, 1);
  }
  estimateLessThan(value) {
    if (this.numBuckets === 0) return 0.5;
    const numVal = toNum2(value);
    if (numVal === null) return 0.5;
    let lo = 0, hi = this.numBuckets - 1;
    while (lo <= hi) {
      const mid = lo + hi >>> 1;
      if (toNum2(this.boundaries[mid]) < numVal) lo = mid + 1;
      else hi = mid - 1;
    }
    if (lo >= this.numBuckets) return 1;
    if (lo === 0) {
      const bucketMax2 = toNum2(this.boundaries[0]);
      const bucketMin2 = this.numBuckets > 0 ? toNum2(this.boundaries[0]) : bucketMax2;
      const frac2 = bucketMax2 > 0 ? Math.max(0, numVal) / bucketMax2 : 0.5;
      return Math.max(0, Math.min(1, frac2 / this.numBuckets));
    }
    const bucketMin = toNum2(this.boundaries[lo - 1]);
    const bucketMax = toNum2(this.boundaries[lo]);
    const range = bucketMax - bucketMin;
    const frac = range > 0 ? (numVal - bucketMin) / range : 0.5;
    return Math.max(0, Math.min(1, (lo + Math.max(0, Math.min(1, frac))) / this.numBuckets));
  }
  estimateRange(low, high) {
    const fracHigh = this.estimateLessThan(high + (typeof high === "number" ? 0.5 : 0));
    const fracLow = this.estimateLessThan(low);
    return Math.max(1e-3, fracHigh - fracLow);
  }
};
var StatisticsCollector = class _StatisticsCollector {
  static async collect(table) {
    const rowCount = table.rowCount();
    const columnStats = /* @__PURE__ */ new Map();
    for (const colDef of table.getSchema()) {
      const stats = await _StatisticsCollector._collectColumn(table, colDef, rowCount);
      columnStats.set(colDef.name.toUpperCase(), stats);
    }
    const correlations = await _StatisticsCollector._computeCorrelations(table, columnStats);
    return new TableStatistics(rowCount, columnStats, correlations);
  }
  static async _collectColumn(table, colDef, totalRows) {
    const values = [];
    const valueCounts = /* @__PURE__ */ new Map();
    let nullCount = 0;
    let min2 = null;
    let max2 = null;
    let totalWidth = 0;
    let totalLength = 0;
    let stringCount = 0;
    const colIdx = table.getColumnIndex(colDef.name);
    const isNumeric2 = ["INT32", "INT64", "FLOAT64", "DECIMAL", "DATE"].includes(colDef.dataType);
    const isString = ["VARCHAR", "TEXT", "STRING"].includes(colDef.dataType);
    for await (const chunk of table.scan()) {
      for (let i = 0; i < chunk.size; i++) {
        const val = chunk.getValue(i, colIdx);
        if (val === null || val === void 0) {
          nullCount++;
          continue;
        }
        const normalized = typeof val === "bigint" ? Number(val) : val;
        values.push(normalized);
        const key = String(normalized);
        valueCounts.set(key, (valueCounts.get(key) || 0) + 1);
        if (min2 === null || val < min2) min2 = val;
        if (max2 === null || val > max2) max2 = val;
        if (typeof val === "string") {
          totalWidth += val.length * 2;
          totalLength += val.length;
          stringCount++;
        } else if (typeof val === "bigint") {
          totalWidth += 8;
        } else {
          totalWidth += 4;
        }
      }
    }
    const ndv = new Set(values.map(String)).size;
    const nullFraction = totalRows > 0 ? nullCount / totalRows : 0;
    const avgWidth = values.length > 0 ? Math.ceil(totalWidth / values.length) : 8;
    const avgLength = stringCount > 0 ? totalLength / stringCount : null;
    let histogram = null;
    if (isNumeric2 && values.length > 0) {
      histogram = _StatisticsCollector._buildHistogram(values, totalRows);
    }
    let mcv = null;
    if (valueCounts.size > 0) {
      mcv = _StatisticsCollector._buildMCV(valueCounts, values.length);
    }
    return new ColumnStatistics({ ndv, min: min2, max: max2, nullFraction, histogram, mcv, avgWidth, avgLength });
  }
  static _buildHistogram(values, totalRows) {
    const sorted = [...values].sort((a, b) => {
      const na = typeof a === "bigint" ? Number(a) : a;
      const nb = typeof b === "bigint" ? Number(b) : b;
      return na - nb;
    });
    const numBuckets = Math.min(HISTOGRAM_BUCKETS, Math.max(1, Math.floor(sorted.length / 4)));
    const boundaries = [];
    const step = Math.max(1, Math.floor(sorted.length / numBuckets));
    for (let i = 1; i <= numBuckets; i++) {
      const idx = Math.min(i * step - 1, sorted.length - 1);
      boundaries.push(sorted[idx]);
    }
    return new EquiDepthHistogram(boundaries, totalRows);
  }
  static _buildMCV(valueCounts, nonNullCount) {
    const entries = [...valueCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, MCV_COUNT);
    if (entries.length === 0) return null;
    return {
      values: entries.map((e) => e[0]),
      frequencies: entries.map((e) => e[1] / nonNullCount)
    };
  }
};
var CORRELATION_SAMPLE_SIZE = 1e3;
var CORRELATION_THRESHOLD = 0.3;
StatisticsCollector._computeCorrelations = async function(table, columnStats) {
  const correlations = /* @__PURE__ */ new Map();
  const numericCols = [];
  const schema = table.getSchema();
  for (const colDef of schema) {
    if (["INT32", "INT64", "FLOAT64", "DECIMAL"].includes(colDef.dataType)) {
      numericCols.push({ name: colDef.name.toUpperCase(), idx: table.getColumnIndex(colDef.name) });
    }
  }
  if (numericCols.length < 2) return correlations;
  const samples = /* @__PURE__ */ new Map();
  for (const col2 of numericCols) samples.set(col2.name, []);
  let rowsSampled = 0;
  const totalRows = table.rowCount();
  const sampleRate = Math.min(1, CORRELATION_SAMPLE_SIZE / Math.max(totalRows, 1));
  let deterministicCounter = 0;
  for await (const chunk of table.scan()) {
    for (let i = 0; i < chunk.size && rowsSampled < CORRELATION_SAMPLE_SIZE; i++) {
      deterministicCounter++;
      if (sampleRate < 1 && deterministicCounter * sampleRate % 1 >= sampleRate) continue;
      let hasNull = false;
      for (const col2 of numericCols) {
        const val = chunk.getValue(i, col2.idx);
        if (val === null || val === void 0) {
          hasNull = true;
          break;
        }
      }
      if (hasNull) continue;
      for (const col2 of numericCols) {
        const val = chunk.getValue(i, col2.idx);
        samples.get(col2.name).push(typeof val === "bigint" ? Number(val) : val);
      }
      rowsSampled++;
    }
    if (rowsSampled >= CORRELATION_SAMPLE_SIZE) break;
  }
  if (rowsSampled < 2) return correlations;
  for (let i = 0; i < numericCols.length; i++) {
    for (let j = i + 1; j < numericCols.length; j++) {
      const colA = numericCols[i].name;
      const colB = numericCols[j].name;
      const valsA = samples.get(colA);
      const valsB = samples.get(colB);
      const corr = pearsonCorrelation(valsA, valsB);
      if (Math.abs(corr) >= CORRELATION_THRESHOLD) {
        const key = colA < colB ? `${colA}:${colB}` : `${colB}:${colA}`;
        correlations.set(key, corr);
      }
    }
  }
  return correlations;
};
function pearsonCorrelation(xs, ys) {
  const n = xs.length;
  if (n < 2) return 0;
  let sumX = 0, sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += xs[i];
    sumY += ys[i];
  }
  const meanX = sumX / n;
  const meanY = sumY / n;
  let cov = 0, varX = 0, varY = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - meanX;
    const dy = ys[i] - meanY;
    cov += dx * dy;
    varX += dx * dx;
    varY += dy * dy;
  }
  const denom = Math.sqrt(varX * varY);
  return denom > 0 ? cov / denom : 0;
}
function toNum2(v) {
  if (v === null || v === void 0) return null;
  if (typeof v === "bigint") return Number(v);
  if (typeof v === "number") return v;
  return null;
}

// src/optimizer/passes/expression-simplifier.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var ExpressionSimplifier = class extends OptimizationPass {
  get name() {
    return "ExpressionSimplifier";
  }
  apply(plan) {
    const rewriter = new SimplifierRewriter();
    return rewriter.rewrite(plan);
  }
};
var SimplifierRewriter = class extends PlanRewriter {
  rewriteFilter(node) {
    const child = this.rewrite(node.children[0]);
    const simplifiedCond = simplifyExpression(node.condition);
    if (simplifiedCond.kind === BoundExprKind.LITERAL && simplifiedCond.value === true) {
      return child;
    }
    if (simplifiedCond.kind === BoundExprKind.LITERAL && simplifiedCond.value === false) {
      return { type: PlanNodeType.EMPTY, children: [child] };
    }
    if (simplifiedCond !== node.condition || child !== node.children[0]) {
      return { ...node, condition: simplifiedCond, children: [child] };
    }
    return node;
  }
  rewriteProject(node) {
    const child = this.rewrite(node.children[0]);
    let changed = false;
    const newExprs = node.expressions.map((expr2) => {
      const s = simplifyExpression(expr2);
      if (s !== expr2) changed = true;
      return s;
    });
    if (changed || child !== node.children[0]) {
      return { ...node, expressions: newExprs, children: [child] };
    }
    return node;
  }
  rewriteJoin(node) {
    const newChildren = node.children.map((c) => this.rewrite(c));
    let changed = newChildren.some((c, i) => c !== node.children[i]);
    let newCond = node.condition;
    if (node.condition) {
      newCond = simplifyExpression(node.condition);
      if (newCond !== node.condition) changed = true;
    }
    if (changed) {
      return { ...node, condition: newCond, children: newChildren };
    }
    return node;
  }
};
function simplifyExpression(expr2) {
  if (!expr2) return expr2;
  switch (expr2.kind) {
    case BoundExprKind.BINARY: {
      const left = simplifyExpression(expr2.left);
      const right = simplifyExpression(expr2.right);
      const op = expr2.op.toUpperCase();
      if (op === "AND") {
        if (isLiteral(left, false) || isLiteral(right, false)) return BoundLiteral(false, "BOOLEAN");
        if (isLiteral(left, true)) return right;
        if (isLiteral(right, true)) return left;
        if (exprEquals(left, right)) return left;
      } else if (op === "OR") {
        if (isLiteral(left, true) || isLiteral(right, true)) return BoundLiteral(true, "BOOLEAN");
        if (isLiteral(left, false)) return right;
        if (isLiteral(right, false)) return left;
        if (exprEquals(left, right)) return left;
        const factored = factorCommonConjuncts(left, right);
        if (factored) return simplifyExpression(factored);
      } else {
        if (op === "+" || op === "-") {
          if (isLiteral(right, 0)) return left;
          if (op === "+" && isLiteral(left, 0)) return right;
        }
        if (op === "*") {
          if (isLiteral(right, 1)) return left;
          if (isLiteral(left, 1)) return right;
        }
        if (op === "/") {
          if (isLiteral(right, 1)) return left;
        }
        if (left.kind === BoundExprKind.LITERAL && right.kind === BoundExprKind.LITERAL) {
          const lVal = left.value;
          const rVal = right.value;
          if (lVal !== null && rVal !== null) {
            try {
              if (op === "+") return BoundLiteral(lVal + rVal, typeof (lVal + rVal) === "number" ? "FLOAT64" : expr2.dataType);
              if (op === "-") return BoundLiteral(lVal - rVal, typeof (lVal - rVal) === "number" ? "FLOAT64" : expr2.dataType);
              if (op === "*") return BoundLiteral(lVal * rVal, typeof (lVal * rVal) === "number" ? "FLOAT64" : expr2.dataType);
              if (op === "/") return BoundLiteral(lVal / rVal, "FLOAT64");
              if (op === "=") return BoundLiteral(lVal === rVal, "BOOLEAN");
              if (op === "!=") return BoundLiteral(lVal !== rVal, "BOOLEAN");
              if (op === ">") return BoundLiteral(lVal > rVal, "BOOLEAN");
              if (op === ">=") return BoundLiteral(lVal >= rVal, "BOOLEAN");
              if (op === "<") return BoundLiteral(lVal < rVal, "BOOLEAN");
              if (op === "<=") return BoundLiteral(lVal <= rVal, "BOOLEAN");
            } catch (e) {
            }
          }
        }
      }
      if (left !== expr2.left || right !== expr2.right) {
        return { ...expr2, left, right };
      }
      return expr2;
    }
    case BoundExprKind.UNARY: {
      const operand = simplifyExpression(expr2.operand);
      const op = expr2.op.toUpperCase();
      if (op === "NOT") {
        if (isLiteral(operand, true)) return BoundLiteral(false, "BOOLEAN");
        if (isLiteral(operand, false)) return BoundLiteral(true, "BOOLEAN");
        if (operand.kind === BoundExprKind.UNARY && operand.op.toUpperCase() === "NOT") {
          return operand.operand;
        }
      }
      if (operand !== expr2.operand) {
        return { ...expr2, operand };
      }
      return expr2;
    }
    case BoundExprKind.FUNCTION: {
      const args = expr2.args.map(simplifyExpression);
      const changed = args.some((a, i) => a !== expr2.args[i]);
      if (changed) {
        return { ...expr2, args };
      }
      return expr2;
    }
    case BoundExprKind.CASE: {
      const whenClauses = [];
      let changed = false;
      for (const wc of expr2.whenClauses) {
        const cond = simplifyExpression(wc.condition);
        const result = simplifyExpression(wc.result);
        if (cond !== wc.condition || result !== wc.result) changed = true;
        if (isLiteral(cond, false)) {
          changed = true;
          continue;
        }
        if (isLiteral(cond, true)) {
          return result;
        }
        whenClauses.push({ condition: cond, result });
      }
      const elseExpr = expr2.elseExpr ? simplifyExpression(expr2.elseExpr) : null;
      if (elseExpr !== expr2.elseExpr) changed = true;
      if (whenClauses.length === 0 && elseExpr) return elseExpr;
      if (whenClauses.length === 0 && !elseExpr) return BoundLiteral(null, expr2.resultType);
      if (changed) {
        return { ...expr2, whenClauses, elseExpr };
      }
      return expr2;
    }
    case BoundExprKind.CAST: {
      const inner = simplifyExpression(expr2.expr);
      if (inner.kind === BoundExprKind.LITERAL && inner.value !== null) {
        const folded = foldCast(inner.value, expr2.targetType);
        if (folded !== void 0) return BoundLiteral(folded, expr2.targetType);
      }
      if (inner !== expr2.expr) return { ...expr2, expr: inner };
      return expr2;
    }
    case BoundExprKind.EXTRACT: {
      const source = simplifyExpression(expr2.source);
      if (source.kind === BoundExprKind.LITERAL && source.value instanceof Date) {
        const extracted = extractFromDate(source.value, expr2.field);
        if (extracted !== null) return BoundLiteral(extracted, "INT32");
      }
      if (source !== expr2.source) return { ...expr2, source };
      return expr2;
    }
  }
  return expr2;
}
function isLiteral(expr2, val) {
  return expr2 && expr2.kind === BoundExprKind.LITERAL && expr2.value === val;
}
function exprEquals(e1, e2) {
  if (e1 === e2) return true;
  if (!e1 || !e2) return false;
  if (e1.kind !== e2.kind) return false;
  if (e1.kind === BoundExprKind.COLUMN_REF) {
    return e1.tableAlias === e2.tableAlias && e1.columnName === e2.columnName;
  }
  if (e1.kind === BoundExprKind.LITERAL) {
    return e1.value === e2.value;
  }
  return exprKey(e1) === exprKey(e2);
}
function factorCommonConjuncts(left, right) {
  const leftConjuncts = splitConjuncts2(left);
  const rightConjuncts = splitConjuncts2(right);
  const rightByKey = new Map(rightConjuncts.map((expr2) => [exprKey(expr2), expr2]));
  const common = [];
  const leftRest = [];
  const commonKeys = /* @__PURE__ */ new Set();
  for (const expr2 of leftConjuncts) {
    const key = exprKey(expr2);
    if (rightByKey.has(key)) {
      common.push(expr2);
      commonKeys.add(key);
    } else {
      leftRest.push(expr2);
    }
  }
  if (common.length === 0) return null;
  const rightRest = rightConjuncts.filter((expr2) => !commonKeys.has(exprKey(expr2)));
  const leftRemainder = combineConjuncts2(leftRest) || BoundLiteral(true, "BOOLEAN");
  const rightRemainder = combineConjuncts2(rightRest) || BoundLiteral(true, "BOOLEAN");
  const residualOr = {
    kind: BoundExprKind.BINARY,
    op: "OR",
    left: leftRemainder,
    right: rightRemainder,
    resultType: "BOOLEAN"
  };
  return combineConjuncts2([...common, residualOr]);
}
function splitConjuncts2(expr2) {
  if (!expr2) return [];
  if (expr2.kind === BoundExprKind.BINARY && expr2.op.toUpperCase() === "AND") {
    return [...splitConjuncts2(expr2.left), ...splitConjuncts2(expr2.right)];
  }
  return [expr2];
}
function combineConjuncts2(exprs) {
  if (exprs.length === 0) return null;
  return exprs.reduce((acc, expr2) => acc ? {
    kind: BoundExprKind.BINARY,
    op: "AND",
    left: acc,
    right: expr2,
    resultType: "BOOLEAN"
  } : expr2, null);
}
function foldCast(value, targetType) {
  const type = (targetType || "").toUpperCase();
  try {
    if (type.includes("INT")) return Math.trunc(Number(value));
    if (type.includes("FLOAT") || type.includes("DOUBLE") || type.includes("DECIMAL") || type.includes("NUMERIC")) return Number(value);
    if (type.includes("VARCHAR") || type.includes("TEXT") || type.includes("CHAR")) return String(value);
    if (type.includes("BOOL")) return Boolean(value);
  } catch (_) {
  }
  return void 0;
}
function extractFromDate(date, field) {
  const f = (field || "").toUpperCase();
  if (f === "YEAR") return date.getFullYear();
  if (f === "MONTH") return date.getMonth() + 1;
  if (f === "DAY") return date.getDate();
  if (f === "HOUR") return date.getHours();
  if (f === "MINUTE") return date.getMinutes();
  if (f === "SECOND") return date.getSeconds();
  return null;
}
function exprKey(expr2) {
  if (!expr2 || typeof expr2 !== "object") return String(expr2);
  switch (expr2.kind) {
    case BoundExprKind.COLUMN_REF:
      return `COL:${expr2.tableAlias || ""}.${expr2.columnName}`;
    case BoundExprKind.LITERAL:
      return `LIT:${String(expr2.value)}`;
    case BoundExprKind.BINARY:
      return `BIN:${expr2.op}:${exprKey(expr2.left)}:${exprKey(expr2.right)}`;
    case BoundExprKind.LIKE:
      return `LIKE:${expr2.negated}:${exprKey(expr2.expr)}:${exprKey(expr2.pattern)}`;
    case BoundExprKind.IN_LIST:
      return `IN:${expr2.negated}:${exprKey(expr2.expr)}:${Array.isArray(expr2.list) ? expr2.list.map(exprKey).join(",") : exprKey(expr2.list)}`;
    case BoundExprKind.BETWEEN:
      return `BETWEEN:${expr2.negated}:${exprKey(expr2.expr)}:${exprKey(expr2.low)}:${exprKey(expr2.high)}`;
    default:
      return JSON.stringify(expr2);
  }
}

// src/optimizer/passes/outer-to-inner.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var OuterToInnerJoin = class extends OptimizationPass {
  get name() {
    return "OuterToInnerJoin";
  }
  apply(plan) {
    const rewriter = new OuterToInnerRewriter();
    return rewriter.rewrite(plan);
  }
};
var OuterToInnerRewriter = class extends PlanRewriter {
  rewriteFilter(node) {
    let child = this.rewrite(node.children[0]);
    if (child.type === PlanNodeType.JOIN && (child.joinType === JoinType.LEFT || child.joinType === JoinType.FULL || child.joinType === JoinType.RIGHT || child.joinType === JoinType.SINGLE)) {
      const leftRefs = getPlanRefs(child.children[0]);
      const rightRefs = getPlanRefs(child.children[1]);
      const predicates = splitConjuncts3(node.condition);
      let rejectRightNulls = false;
      let rejectLeftNulls = false;
      for (const pred of predicates) {
        if (isNullRejecting(pred, rightRefs)) {
          rejectRightNulls = true;
        }
        if (isNullRejecting(pred, leftRefs)) {
          rejectLeftNulls = true;
        }
      }
      let newJoinType = child.joinType;
      if (child.joinType === JoinType.LEFT && rejectRightNulls) {
        newJoinType = JoinType.INNER;
      } else if (child.joinType === JoinType.SINGLE && rejectRightNulls) {
        newJoinType = JoinType.INNER;
      } else if (child.joinType === JoinType.RIGHT && rejectLeftNulls) {
        newJoinType = JoinType.INNER;
      } else if (child.joinType === JoinType.FULL) {
        if (rejectLeftNulls && rejectRightNulls) newJoinType = JoinType.INNER;
        else if (rejectLeftNulls) newJoinType = JoinType.LEFT;
        else if (rejectRightNulls) newJoinType = JoinType.RIGHT;
      }
      if (newJoinType !== child.joinType) {
        child = { ...child, joinType: newJoinType };
      }
    }
    if (child !== node.children[0]) {
      return { ...node, children: [child] };
    }
    return node;
  }
};
function getPlanRefs(planNode) {
  const refs = { aliases: /* @__PURE__ */ new Set(), columns: /* @__PURE__ */ new Set() };
  addOutputRefs3(planNode, refs);
  refs.aliases.delete("");
  refs.columns.delete("");
  return refs;
}
function addOutputRefs3(node, refs) {
  if (!node) return;
  if (node.type === PlanNodeType.SCAN) {
    refs.aliases.add((node.alias || node.table || "").toUpperCase());
    for (const col2 of node.columns || []) {
      refs.columns.add((col2.name || col2.columnName || "").toUpperCase());
    }
    return;
  }
  if (node.type === PlanNodeType.CTE_SCAN) {
    refs.aliases.add((node.alias || node.cteName || "").toUpperCase());
    return;
  }
  if (node.type === PlanNodeType.PROJECT) {
    for (const expr2 of node.expressions || []) {
      refs.columns.add((expr2.outputName || expr2.alias || expr2.name || expr2.columnName || "").toUpperCase());
    }
    return;
  }
  if (node.type === PlanNodeType.AGGREGATE) {
    for (const expr2 of node.groupBy || []) {
      refs.columns.add((expr2.outputName || expr2.alias || expr2.name || expr2.columnName || "").toUpperCase());
    }
    for (const agg of node.aggregates || []) {
      refs.columns.add((agg.outputName || agg.alias || agg.name || "").toUpperCase());
    }
    return;
  }
  if (node.type === PlanNodeType.JOIN || node.type === PlanNodeType.UNION) {
    for (const child of getChildren(node)) addOutputRefs3(child, refs);
    return;
  }
  if (node.children?.[0]) addOutputRefs3(node.children[0], refs);
}
function splitConjuncts3(expr2) {
  if (!expr2) return [];
  if (expr2.kind === BoundExprKind.BINARY && expr2.op.toUpperCase() === "AND") {
    return [...splitConjuncts3(expr2.left), ...splitConjuncts3(expr2.right)];
  }
  return [expr2];
}
function isNullRejecting(expr2, nullSupplyingRefs) {
  const result = evaluateWithNulls(expr2, nullSupplyingRefs);
  return result === false || result === null;
}
function evaluateWithNulls(expr2, nullRefs) {
  if (!expr2) return void 0;
  switch (expr2.kind) {
    case BoundExprKind.LITERAL:
      return expr2.value;
    case BoundExprKind.COLUMN_REF:
      if (expr2.tableAlias && nullRefs.aliases.has(expr2.tableAlias.toUpperCase())) {
        return null;
      }
      if (!expr2.tableAlias && nullRefs.columns.has((expr2.columnName || "").toUpperCase())) {
        return null;
      }
      return "UNKNOWN";
    case BoundExprKind.BINARY: {
      const left = evaluateWithNulls(expr2.left, nullRefs);
      const right = evaluateWithNulls(expr2.right, nullRefs);
      const op = expr2.op.toUpperCase();
      if (op === "AND") {
        if (left === false || right === false) return false;
        if (left === null || right === null) return null;
        if (left === "UNKNOWN" || right === "UNKNOWN") return "UNKNOWN";
        return true;
      }
      if (op === "OR") {
        if (left === true || right === true) return true;
        if (left === "UNKNOWN" || right === "UNKNOWN") return "UNKNOWN";
        if (left === null && right === null) return null;
        return false;
      }
      if (left === null || right === null) return null;
      return "UNKNOWN";
    }
    case BoundExprKind.UNARY: {
      const operand = evaluateWithNulls(expr2.operand, nullRefs);
      if (operand === null) return null;
      return "UNKNOWN";
    }
    case BoundExprKind.FUNCTION:
    case BoundExprKind.EXTRACT:
    case BoundExprKind.CAST:
    case BoundExprKind.INTERVAL:
    case BoundExprKind.CASE:
    case BoundExprKind.AGGREGATE:
      return "UNKNOWN";
    case BoundExprKind.IS_NULL: {
      const operand = evaluateWithNulls(expr2.operand, nullRefs);
      if (operand === null) {
        return true;
      }
      return "UNKNOWN";
    }
  }
  return "UNKNOWN";
}

// src/optimizer/passes/limit-pushdown.js
init_buffer_shim();
init_logical_plan();
var LimitPushdown = class extends OptimizationPass {
  get name() {
    return "LimitPushdown";
  }
  apply(plan) {
    const rewriter = new LimitPushdownRewriter();
    return rewriter.rewrite(plan);
  }
};
var LimitPushdownRewriter = class extends PlanRewriter {
  rewriteLimit(node) {
    const child = this.rewrite(node.children[0]);
    if (child.type === PlanNodeType.PROJECT) {
      const newLimit = { ...node, children: [child.children[0]] };
      const newProject = { ...child, children: [newLimit] };
      const optimizedLimit = this.rewrite(newLimit);
      return { ...newProject, children: [optimizedLimit] };
    }
    if (child.type === PlanNodeType.UNION && child.all) {
      const leftLimit = { ...node, children: [child.children[0]] };
      const rightLimit = { ...node, children: [child.children[1]] };
      const newUnion = {
        ...child,
        children: [
          this.rewrite(leftLimit),
          this.rewrite(rightLimit)
        ]
      };
      return { ...node, children: [newUnion] };
    }
    if (child.type === PlanNodeType.SORT) {
      const newSort = { ...child, limit: node.count, offset: node.offset || 0 };
      return { ...node, children: [newSort] };
    }
    if (child.type === PlanNodeType.AGGREGATE && child.groupBy && child.groupBy.length > 0) {
      const newAgg = { ...child, _limitHint: node.count + (node.offset || 0) };
      return { ...node, children: [newAgg] };
    }
    if (child !== node.children[0]) {
      return { ...node, children: [child] };
    }
    return node;
  }
};

// src/optimizer/passes/having-pushdown.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var HavingPushdown = class extends OptimizationPass {
  get name() {
    return "HavingPushdown";
  }
  apply(plan) {
    const rewriter = new HavingPushdownRewriter();
    return rewriter.rewrite(plan);
  }
};
var HavingPushdownRewriter = class extends PlanRewriter {
  rewriteFilter(node) {
    let child = this.rewrite(node.children[0]);
    if (child.type === PlanNodeType.AGGREGATE) {
      const predicates = splitConjuncts4(node.condition);
      const pushable = [];
      const unpushable = [];
      for (const pred of predicates) {
        if (containsAggregate2(pred)) {
          unpushable.push(pred);
        } else {
          pushable.push(pred);
        }
      }
      if (pushable.length > 0) {
        const aggChild = child.children[0];
        const pushedCond = combineConjuncts3(pushable);
        const newBottomFilter = {
          type: PlanNodeType.FILTER,
          condition: pushedCond,
          children: [aggChild]
        };
        child = { ...child, children: [newBottomFilter] };
        if (unpushable.length === 0) {
          return child;
        } else {
          return { ...node, condition: combineConjuncts3(unpushable), children: [child] };
        }
      }
    }
    if (child !== node.children[0]) {
      return { ...node, children: [child] };
    }
    return node;
  }
};
function splitConjuncts4(expr2) {
  if (!expr2) return [];
  if (expr2.kind === BoundExprKind.BINARY && expr2.op?.toUpperCase() === "AND") {
    return [...splitConjuncts4(expr2.left), ...splitConjuncts4(expr2.right)];
  }
  return [expr2];
}
function combineConjuncts3(exprs) {
  if (!exprs || exprs.length === 0) return null;
  let result = exprs[0];
  for (let i = 1; i < exprs.length; i++) {
    result = {
      kind: BoundExprKind.BINARY,
      op: "AND",
      left: result,
      right: exprs[i],
      resultType: "BOOLEAN"
    };
  }
  return result;
}
function containsAggregate2(expr2) {
  if (!expr2) return false;
  if (expr2.kind === BoundExprKind.AGGREGATE) return true;
  if (expr2.kind === BoundExprKind.BINARY) {
    return containsAggregate2(expr2.left) || containsAggregate2(expr2.right);
  }
  if (expr2.kind === BoundExprKind.UNARY) {
    return containsAggregate2(expr2.operand);
  }
  if (expr2.kind === BoundExprKind.FUNCTION || expr2.kind === BoundExprKind.CASE) {
    if (expr2.args) {
      return expr2.args.some(containsAggregate2);
    }
    if (expr2.whenClauses) {
      for (const wc of expr2.whenClauses) {
        if (containsAggregate2(wc.condition) || containsAggregate2(wc.result)) return true;
      }
    }
    if (expr2.operand && containsAggregate2(expr2.operand)) return true;
    if (expr2.elseExpr && containsAggregate2(expr2.elseExpr)) return true;
  }
  return false;
}

// src/optimizer/passes/empty-propagation.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var EmptyPropagation = class extends OptimizationPass {
  get name() {
    return "EmptyPropagation";
  }
  apply(plan) {
    const rewriter = new EmptyPropagationRewriter();
    return rewriter.rewrite(plan);
  }
};
var EmptyPropagationRewriter = class extends PlanRewriter {
  rewriteDefault(node) {
    const newNode = this.rewriteChildren(node);
    if (newNode.children && newNode.children.length === 1 && newNode.children[0].type === PlanNodeType.EMPTY) {
      if (newNode.type === PlanNodeType.AGGREGATE && (!newNode.groupBy || newNode.groupBy.length === 0)) {
        return newNode;
      }
      return newNode.children[0];
    }
    return newNode;
  }
  rewriteFilter(node) {
    const newNode = this.rewriteChildren(node);
    if (newNode.children[0].type === PlanNodeType.EMPTY) {
      return newNode.children[0];
    }
    if (newNode.condition && newNode.condition.kind === BoundExprKind.LITERAL && newNode.condition.value === false) {
      return { type: PlanNodeType.EMPTY, children: [newNode.children[0]] };
    }
    return newNode;
  }
  rewriteLimit(node) {
    const newNode = this.rewriteChildren(node);
    if (newNode.children[0].type === PlanNodeType.EMPTY) {
      return newNode.children[0];
    }
    if (newNode.count === 0) {
      return { type: PlanNodeType.EMPTY, children: [newNode.children[0]] };
    }
    return newNode;
  }
  rewriteJoin(node) {
    const newNode = this.rewriteChildren(node);
    const left = newNode.children[0];
    const right = newNode.children[1];
    const leftEmpty = left.type === PlanNodeType.EMPTY;
    const rightEmpty = right.type === PlanNodeType.EMPTY;
    if (newNode.joinType === JoinType.INNER || newNode.joinType === JoinType.CROSS) {
      if (leftEmpty || rightEmpty) {
        return { type: PlanNodeType.EMPTY, children: [newNode] };
      }
    } else if (newNode.joinType === JoinType.LEFT) {
      if (leftEmpty) {
        return { type: PlanNodeType.EMPTY, children: [newNode] };
      }
      if (rightEmpty) {
        return newNode;
      }
    } else if (newNode.joinType === JoinType.FULL) {
      if (leftEmpty && rightEmpty) {
        return { type: PlanNodeType.EMPTY, children: [newNode] };
      }
    }
    return newNode;
  }
  rewriteUnion(node) {
    const newNode = this.rewriteChildren(node);
    const left = newNode.children[0];
    const right = newNode.children[1];
    const leftEmpty = left.type === PlanNodeType.EMPTY;
    const rightEmpty = right.type === PlanNodeType.EMPTY;
    if (leftEmpty && rightEmpty) {
      return { type: PlanNodeType.EMPTY, children: [newNode] };
    }
    if (leftEmpty) return right;
    if (rightEmpty) return left;
    return newNode;
  }
};

// src/optimizer/passes/node-merge.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var NodeMerge = class extends OptimizationPass {
  get name() {
    return "NodeMerge";
  }
  apply(plan) {
    const rewriter = new NodeMergeRewriter();
    return rewriter.rewrite(plan);
  }
};
var NodeMergeRewriter = class extends PlanRewriter {
  rewriteFilter(node) {
    let child = this.rewrite(node.children[0]);
    if (child.type === PlanNodeType.FILTER) {
      const mergedCond = {
        kind: BoundExprKind.BINARY,
        op: "AND",
        left: node.condition,
        right: child.condition,
        resultType: "BOOLEAN"
      };
      child = child.children[0];
      return { ...node, condition: mergedCond, children: [child] };
    }
    if (child !== node.children[0]) {
      return { ...node, children: [child] };
    }
    return node;
  }
  rewriteProject(node) {
    const child = this.rewrite(node.children[0]);
    if (child.type === PlanNodeType.PROJECT && sameProjectExpressions(node.expressions, child.expressions)) {
      return { ...node, children: [child.children[0]] };
    }
    if (child !== node.children[0]) {
      return { ...node, children: [child] };
    }
    return node;
  }
  rewriteLimit(node) {
    const child = this.rewrite(node.children[0]);
    if (child.type === PlanNodeType.LIMIT) {
      const mergedCount = Math.min(node.count, child.count);
      return { ...node, count: mergedCount, children: [child.children[0]] };
    }
    if (child !== node.children[0]) {
      return { ...node, children: [child] };
    }
    return node;
  }
};
function sameProjectExpressions(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right)) return false;
  if (left.length !== right.length) return false;
  return left.every((expr2, i) => exprEqualsIgnoringOutput(expr2, right[i]));
}
function exprEqualsIgnoringOutput(left, right) {
  if (left === right) return true;
  if (!left || !right) return false;
  if (typeof left !== "object" || typeof right !== "object") return left === right;
  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false;
    return left.every((item, i) => exprEqualsIgnoringOutput(item, right[i]));
  }
  const leftKeys = Object.keys(left).filter(isSemanticKey).sort();
  const rightKeys = Object.keys(right).filter(isSemanticKey).sort();
  if (leftKeys.length !== rightKeys.length) return false;
  for (let i = 0; i < leftKeys.length; i++) {
    if (leftKeys[i] !== rightKeys[i]) return false;
    if (!exprEqualsIgnoringOutput(left[leftKeys[i]], right[rightKeys[i]])) return false;
  }
  return true;
}
function isSemanticKey(key) {
  return key !== "outputName" && key !== "alias";
}

// src/optimizer/passes/predicate-inference.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var PredicateInference = class extends OptimizationPass {
  get name() {
    return "PredicateInference";
  }
  apply(plan) {
    const rewriter = new InferenceRewriter();
    return rewriter.rewrite(plan);
  }
};
var InferenceRewriter = class extends PlanRewriter {
  rewriteFilter(node) {
    const child = this.rewrite(node.children[0]);
    const preds = splitConjuncts(node.condition);
    const inferred = inferNewPredicates(preds);
    if (inferred.length === 0) {
      if (child !== node.children[0]) {
        return { ...node, children: [child] };
      }
      return node;
    }
    const allPreds = [...preds, ...inferred];
    return LogicalFilter(combineConjuncts(allPreds), child);
  }
  rewriteJoin(node) {
    const newNode = this.rewriteChildren(node);
    if (newNode.joinType !== JoinType.INNER || !newNode.condition) return newNode;
    const preds = splitConjuncts(newNode.condition);
    const leftTables = collectTables(newNode.children[0]);
    const rightTables = collectTables(newNode.children[1]);
    const allPreds = [...preds];
    collectFiltersAbove(newNode.children[0], allPreds);
    collectFiltersAbove(newNode.children[1], allPreds);
    const inferred = inferNewPredicates(allPreds);
    if (inferred.length === 0) return newNode;
    const leftPush = [];
    const rightPush = [];
    for (const pred of inferred) {
      const refs = collectTableRefs2(pred);
      if (refs.length === 0) continue;
      const leftOnly = refs.every((r) => leftTables.has(r));
      const rightOnly = refs.every((r) => rightTables.has(r));
      if (leftOnly) leftPush.push(pred);
      else if (rightOnly) rightPush.push(pred);
    }
    let left = newNode.children[0];
    let right = newNode.children[1];
    if (leftPush.length > 0) {
      left = LogicalFilter(combineConjuncts(leftPush), left);
    }
    if (rightPush.length > 0) {
      right = LogicalFilter(combineConjuncts(rightPush), right);
    }
    if (left !== newNode.children[0] || right !== newNode.children[1]) {
      return { ...newNode, children: [left, right] };
    }
    return newNode;
  }
};
function inferNewPredicates(predicates) {
  const equalities = /* @__PURE__ */ new Map();
  const constants = /* @__PURE__ */ new Map();
  const existingKeys = new Set(predicates.map(predKey));
  const inferred = [];
  for (const pred of predicates) {
    for (const inList of inferInListsFromOr(pred, existingKeys)) {
      inferred.push(inList);
      existingKeys.add(predKey(inList));
    }
    for (const rangePred of inferRangePredicatesFromOr(pred, existingKeys)) {
      inferred.push(rangePred);
      existingKeys.add(predKey(rangePred));
    }
  }
  for (const pred of predicates) {
    if (pred.kind !== BoundExprKind.BINARY || pred.op !== "=") continue;
    const leftIsCol = pred.left?.kind === BoundExprKind.COLUMN_REF;
    const rightIsCol = pred.right?.kind === BoundExprKind.COLUMN_REF;
    const leftIsLit = pred.left?.kind === BoundExprKind.LITERAL;
    const rightIsLit = pred.right?.kind === BoundExprKind.LITERAL;
    if (leftIsCol && rightIsLit) {
      constants.set(colKey(pred.left), pred.right);
    } else if (rightIsCol && leftIsLit) {
      constants.set(colKey(pred.right), pred.left);
    } else if (leftIsCol && rightIsCol) {
      const lk = colKey(pred.left);
      const rk = colKey(pred.right);
      if (!equalities.has(lk)) equalities.set(lk, /* @__PURE__ */ new Set());
      if (!equalities.has(rk)) equalities.set(rk, /* @__PURE__ */ new Set());
      equalities.get(lk).add(rk);
      equalities.get(rk).add(lk);
    }
  }
  for (const [colK, litExpr] of constants) {
    const equivCols = equalities.get(colK);
    if (!equivCols) continue;
    for (const eqColK of equivCols) {
      if (constants.has(eqColK)) continue;
      const colExpr = findColExpr(predicates, eqColK);
      if (!colExpr) continue;
      const newPred = BoundBinary("=", colExpr, litExpr, "BOOLEAN");
      const key = predKey(newPred);
      if (!existingKeys.has(key)) {
        inferred.push(newPred);
        existingKeys.add(key);
        constants.set(eqColK, litExpr);
      }
    }
  }
  for (const [colK, equivCols] of equalities) {
    const litExpr = constants.get(colK);
    if (!litExpr) continue;
    for (const eqColK of equivCols) {
      if (constants.has(eqColK)) continue;
      const colExpr = findColExpr(predicates, eqColK);
      if (!colExpr) continue;
      const newPred = BoundBinary("=", colExpr, litExpr, "BOOLEAN");
      const key = predKey(newPred);
      if (!existingKeys.has(key)) {
        inferred.push(newPred);
        existingKeys.add(key);
      }
    }
  }
  for (const pred of predicates) {
    if (pred.kind !== BoundExprKind.BINARY) continue;
    if (!["<", ">", "<=", ">=", "<>"].includes(pred.op)) continue;
    const leftIsCol = pred.left?.kind === BoundExprKind.COLUMN_REF;
    const rightIsCol = pred.right?.kind === BoundExprKind.COLUMN_REF;
    const leftIsLit = pred.left?.kind === BoundExprKind.LITERAL;
    const rightIsLit = pred.right?.kind === BoundExprKind.LITERAL;
    if (leftIsCol && rightIsLit) {
      const equivCols = equalities.get(colKey(pred.left));
      if (equivCols) {
        for (const eqColK of equivCols) {
          const colExpr = findColExpr(predicates, eqColK);
          if (!colExpr) continue;
          const newPred = BoundBinary(pred.op, colExpr, pred.right, "BOOLEAN");
          const key = predKey(newPred);
          if (!existingKeys.has(key)) {
            inferred.push(newPred);
            existingKeys.add(key);
          }
        }
      }
    } else if (rightIsCol && leftIsLit) {
      const equivCols = equalities.get(colKey(pred.right));
      if (equivCols) {
        for (const eqColK of equivCols) {
          const colExpr = findColExpr(predicates, eqColK);
          if (!colExpr) continue;
          const newPred = BoundBinary(pred.op, pred.left, colExpr, "BOOLEAN");
          const key = predKey(newPred);
          if (!existingKeys.has(key)) {
            inferred.push(newPred);
            existingKeys.add(key);
          }
        }
      }
    }
  }
  return inferred;
}
function inferInListsFromOr(expr2, existingKeys) {
  if (!expr2 || expr2.kind !== BoundExprKind.BINARY || expr2.op !== "OR") return [];
  const branches = splitOr(expr2);
  if (branches.length < 2) return [];
  const branchMaps = branches.map((branch) => collectBranchConstraints(branch));
  const inferred = [];
  const first = branchMaps[0];
  for (const [key, entry] of first) {
    if (!branchMaps.every((map) => map.has(key))) continue;
    const literals = new Map(entry.literals || []);
    for (let i = 1; i < branchMaps.length; i++) {
      const branchEntry = branchMaps[i].get(key);
      if (!branchEntry.literals || branchEntry.literals.size === 0) {
        literals.clear();
        break;
      }
      for (const [litKey, lit2] of branchEntry.literals) {
        literals.set(litKey, lit2);
      }
    }
    if (literals.size < 2) continue;
    const pred = BoundInList(entry.col, [...literals.values()], false);
    const keyPred = predKey(pred);
    if (!existingKeys.has(keyPred)) inferred.push(pred);
  }
  return inferred;
}
function inferRangePredicatesFromOr(expr2, existingKeys) {
  if (!expr2 || expr2.kind !== BoundExprKind.BINARY || expr2.op !== "OR") return [];
  const branches = splitOr(expr2);
  if (branches.length < 2) return [];
  const branchMaps = branches.map((branch) => collectBranchConstraints(branch));
  const inferred = [];
  const first = branchMaps[0];
  for (const [key, entry] of first) {
    if (!branchMaps.every((map) => map.has(key))) continue;
    const entries = branchMaps.map((map) => map.get(key));
    if (!entries.every((e) => e.lower && e.upper)) continue;
    const lower = entries.reduce((best, e) => compareLiteral(e.lower.literal, best.literal) < 0 ? e.lower : best, entries[0].lower);
    const upper = entries.reduce((best, e) => compareLiteral(e.upper.literal, best.literal) > 0 ? e.upper : best, entries[0].upper);
    const lowerPred = BoundBinary(lower.op, entry.col, lower.literal, "BOOLEAN");
    const upperPred = BoundBinary(upper.op, entry.col, upper.literal, "BOOLEAN");
    for (const pred of [lowerPred, upperPred]) {
      const keyPred = predKey(pred);
      if (!existingKeys.has(keyPred)) inferred.push(pred);
    }
  }
  return inferred;
}
function collectBranchConstraints(branch) {
  const map = /* @__PURE__ */ new Map();
  for (const pred of splitConjuncts(branch)) {
    addEqualityConstraint(map, pred);
    addInListConstraint(map, pred);
    addRangeConstraint(map, pred);
  }
  return map;
}
function ensureConstraint(map, col2) {
  const key = colKey(col2);
  if (!map.has(key)) map.set(key, { col: col2, literals: /* @__PURE__ */ new Map(), lower: null, upper: null });
  return map.get(key);
}
function addEqualityConstraint(map, pred) {
  if (pred.kind !== BoundExprKind.BINARY || pred.op !== "=") return;
  const leftIsCol = pred.left?.kind === BoundExprKind.COLUMN_REF;
  const rightIsCol = pred.right?.kind === BoundExprKind.COLUMN_REF;
  const leftIsLit = pred.left?.kind === BoundExprKind.LITERAL;
  const rightIsLit = pred.right?.kind === BoundExprKind.LITERAL;
  let col2 = null;
  let lit2 = null;
  if (leftIsCol && rightIsLit) {
    col2 = pred.left;
    lit2 = pred.right;
  } else if (rightIsCol && leftIsLit) {
    col2 = pred.right;
    lit2 = pred.left;
  }
  if (!col2 || !lit2) return;
  ensureConstraint(map, col2).literals.set(literalKey(lit2), lit2);
}
function addInListConstraint(map, pred) {
  if (pred.kind !== BoundExprKind.IN_LIST || pred.negated || !Array.isArray(pred.list)) return;
  if (pred.expr?.kind !== BoundExprKind.COLUMN_REF) return;
  if (!pred.list.every((item) => item.kind === BoundExprKind.LITERAL)) return;
  const entry = ensureConstraint(map, pred.expr);
  for (const lit2 of pred.list) entry.literals.set(literalKey(lit2), lit2);
}
function addRangeConstraint(map, pred) {
  if (pred.kind === BoundExprKind.BETWEEN && !pred.negated && pred.expr?.kind === BoundExprKind.COLUMN_REF) {
    if (pred.low?.kind === BoundExprKind.LITERAL) setLower(ensureConstraint(map, pred.expr), ">=", pred.low);
    if (pred.high?.kind === BoundExprKind.LITERAL) setUpper(ensureConstraint(map, pred.expr), "<=", pred.high);
    return;
  }
  if (pred.kind !== BoundExprKind.BINARY) return;
  const leftIsCol = pred.left?.kind === BoundExprKind.COLUMN_REF;
  const rightIsCol = pred.right?.kind === BoundExprKind.COLUMN_REF;
  const leftIsLit = pred.left?.kind === BoundExprKind.LITERAL;
  const rightIsLit = pred.right?.kind === BoundExprKind.LITERAL;
  if (leftIsCol && rightIsLit) {
    const entry = ensureConstraint(map, pred.left);
    if (pred.op === ">=" || pred.op === ">") setLower(entry, pred.op, pred.right);
    if (pred.op === "<=" || pred.op === "<") setUpper(entry, pred.op, pred.right);
  } else if (rightIsCol && leftIsLit) {
    const entry = ensureConstraint(map, pred.right);
    if (pred.op === "<=" || pred.op === "<") setLower(entry, flipRangeOp(pred.op), pred.left);
    if (pred.op === ">=" || pred.op === ">") setUpper(entry, flipRangeOp(pred.op), pred.left);
  }
}
function setLower(entry, op, literal) {
  if (!entry.lower || compareLiteral(literal, entry.lower.literal) > 0) entry.lower = { op, literal };
}
function setUpper(entry, op, literal) {
  if (!entry.upper || compareLiteral(literal, entry.upper.literal) < 0) entry.upper = { op, literal };
}
function flipRangeOp(op) {
  if (op === "<=") return ">=";
  if (op === "<") return ">";
  if (op === ">=") return "<=";
  if (op === ">") return "<";
  return op;
}
function compareLiteral(a, b) {
  if (typeof a.value === "number" && typeof b.value === "number") return a.value - b.value;
  return String(a.value).localeCompare(String(b.value));
}
function splitOr(expr2) {
  if (!expr2) return [];
  if (expr2.kind === BoundExprKind.BINARY && expr2.op === "OR") {
    return [...splitOr(expr2.left), ...splitOr(expr2.right)];
  }
  return [expr2];
}
function colKey(expr2) {
  return `${(expr2.tableAlias || "").toUpperCase()}.${(expr2.columnName || "").toUpperCase()}`;
}
function predKey(pred) {
  if (pred.kind === BoundExprKind.BINARY) {
    return `${pred.op}:${exprKey2(pred.left)}:${exprKey2(pred.right)}`;
  }
  if (pred.kind === BoundExprKind.IN_LIST) {
    return `IN:${exprKey2(pred.expr)}:${pred.list.map(exprKey2).join(",")}:${pred.negated}`;
  }
  return JSON.stringify(pred).slice(0, 80);
}
function exprKey2(expr2) {
  if (!expr2) return "null";
  if (expr2.kind === BoundExprKind.COLUMN_REF) return colKey(expr2);
  if (expr2.kind === BoundExprKind.LITERAL) return `LIT:${expr2.value}`;
  return JSON.stringify(expr2).slice(0, 40);
}
function literalKey(expr2) {
  return `${expr2.dataType || ""}:${String(expr2.value)}`;
}
function findColExpr(predicates, colK) {
  for (const pred of predicates) {
    if (pred.kind !== BoundExprKind.BINARY) continue;
    if (pred.left?.kind === BoundExprKind.COLUMN_REF && colKey(pred.left) === colK) return pred.left;
    if (pred.right?.kind === BoundExprKind.COLUMN_REF && colKey(pred.right) === colK) return pred.right;
  }
  return null;
}
function collectTables(node) {
  const tables = /* @__PURE__ */ new Set();
  function walk(n) {
    if (!n) return;
    if (n.type === PlanNodeType.SCAN) {
      tables.add((n.alias || n.table).toUpperCase());
    }
    for (const child of getChildren(n)) walk(child);
  }
  walk(node);
  return tables;
}
function collectTableRefs2(expr2) {
  const refs = [];
  walkExpr(expr2, (e) => {
    if (e.kind === BoundExprKind.COLUMN_REF && e.tableAlias) {
      refs.push(e.tableAlias.toUpperCase());
    }
  });
  return refs;
}
function collectFiltersAbove(node, preds) {
  if (!node) return;
  if (node.type === PlanNodeType.FILTER) {
    preds.push(...splitConjuncts(node.condition));
  }
}
function walkExpr(expr2, fn) {
  if (!expr2 || typeof expr2 !== "object") return;
  fn(expr2);
  if (expr2.left) walkExpr(expr2.left, fn);
  if (expr2.right) walkExpr(expr2.right, fn);
  if (expr2.operand) walkExpr(expr2.operand, fn);
  if (expr2.expr) walkExpr(expr2.expr, fn);
  if (expr2.low) walkExpr(expr2.low, fn);
  if (expr2.high) walkExpr(expr2.high, fn);
  if (expr2.args) for (const a of expr2.args) walkExpr(a, fn);
  if (Array.isArray(expr2.list)) for (const item of expr2.list) walkExpr(item, fn);
  if (expr2.pattern) walkExpr(expr2.pattern, fn);
  if (expr2.source) walkExpr(expr2.source, fn);
  if (expr2.whenClauses) for (const wc of expr2.whenClauses) {
    walkExpr(wc.condition, fn);
    walkExpr(wc.result, fn);
  }
  if (expr2.elseExpr) walkExpr(expr2.elseExpr, fn);
}

// src/optimizer/passes/sort-elimination.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var SortElimination = class extends OptimizationPass {
  get name() {
    return "SortElimination";
  }
  apply(plan) {
    const rewriter = new SortEliminationRewriter();
    return rewriter.rewrite(plan);
  }
};
var SortEliminationRewriter = class extends PlanRewriter {
  rewriteSort(node) {
    const child = this.rewrite(node.children[0]);
    if (!child._sortedBy || child._sortedBy.length === 0) {
      if (child !== node.children[0]) {
        return { ...node, children: [child] };
      }
      return node;
    }
    const requiredKeys = node.orderKeys.map((ok) => ({
      key: getColumnKey(ok.expr),
      direction: ok.direction || "ASC"
    }));
    if (requiredKeys.some((k) => !k.key)) {
      if (child !== node.children[0]) {
        return { ...node, children: [child] };
      }
      return node;
    }
    const childSorted = child._sortedBy;
    let match = true;
    for (let i = 0; i < requiredKeys.length; i++) {
      if (i >= childSorted.length) {
        match = false;
        break;
      }
      const sortedEntry = childSorted[i];
      const sortedKey = typeof sortedEntry === "object" ? sortedEntry.key : sortedEntry;
      const sortedDir = typeof sortedEntry === "object" ? sortedEntry.direction || "ASC" : "ASC";
      if (!columnMatches(sortedKey, requiredKeys[i].key)) {
        match = false;
        break;
      }
      if (sortedDir.toUpperCase() !== requiredKeys[i].direction.toUpperCase()) {
        match = false;
        break;
      }
    }
    if (match) {
      return child;
    }
    if (child !== node.children[0]) {
      return { ...node, children: [child] };
    }
    return node;
  }
};
function getColumnKey(expr2) {
  if (!expr2) return null;
  if (expr2.kind === BoundExprKind.COLUMN_REF) {
    return `${expr2.tableAlias || ""}.${expr2.columnName}`.toUpperCase();
  }
  return null;
}
function columnMatches(sortedKey, reqKey) {
  if (!sortedKey || !reqKey) return false;
  if (sortedKey === reqKey) return true;
  const sortedCol = sortedKey.split(".").pop();
  const reqCol = reqKey.split(".").pop();
  return sortedCol === reqCol;
}

// src/optimizer/passes/join-elimination.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var JoinElimination = class extends OptimizationPass {
  get name() {
    return "JoinElimination";
  }
  apply(plan) {
    const rewriter = new JoinEliminationRewriter();
    return rewriter.rewrite(plan);
  }
};
var COLUMN_RESTRICTING_PARENTS = /* @__PURE__ */ new Set([PlanNodeType.PROJECT, PlanNodeType.AGGREGATE]);
var JoinEliminationRewriter = class extends PlanRewriter {
  rewriteDefault(node) {
    const newNode = this.rewriteChildren(node);
    if (COLUMN_RESTRICTING_PARENTS.has(newNode.type) && hasLeftJoinChild(newNode)) {
      return tryEliminateLeftJoin(newNode);
    }
    return newNode;
  }
};
function hasLeftJoinChild(node) {
  if (!node.children) return false;
  return node.children.some((c) => c.type === PlanNodeType.JOIN && c.joinType === JoinType.LEFT);
}
function tryEliminateLeftJoin(parent) {
  const newChildren = parent.children.map((child) => {
    if (child.type !== PlanNodeType.JOIN || child.joinType !== JoinType.LEFT) return child;
    const rightTables = collectTableAliases(child.children[1]);
    const rightOutputs = collectOutputNames(child.children[1]);
    const usedAbove = /* @__PURE__ */ new Set();
    collectNodeExprColumns(parent, usedAbove);
    const rightUsed = hasAnyColumnUsed(rightTables, usedAbove) || hasAnyNameUsed(rightOutputs, usedAbove);
    if (!rightUsed) {
      return child.children[0];
    }
    return child;
  });
  const changed = newChildren.some((c, i) => c !== parent.children[i]);
  return changed ? { ...parent, children: newChildren } : parent;
}
function collectNodeExprColumns(node, used) {
  const collectExpr = (expr2) => {
    if (!expr2 || typeof expr2 !== "object") return;
    if (expr2.kind === BoundExprKind.COLUMN_REF) {
      used.add(`${(expr2.tableAlias || "").toUpperCase()}.${(expr2.columnName || "").toUpperCase()}`);
      return;
    }
    for (const val of Object.values(expr2)) {
      if (Array.isArray(val)) {
        for (const item of val) collectExpr(item);
      } else if (val && typeof val === "object") {
        collectExpr(val);
      }
    }
  };
  switch (node.type) {
    case PlanNodeType.PROJECT:
      for (const expr2 of node.expressions) collectExpr(expr2);
      break;
    case PlanNodeType.FILTER:
      collectExpr(node.condition);
      break;
    case PlanNodeType.AGGREGATE:
      if (node.groupBy) for (const g of node.groupBy) collectExpr(g);
      for (const agg of node.aggregates) {
        for (const arg of agg.args) collectExpr(arg);
      }
      break;
    case PlanNodeType.SORT:
      for (const ok of node.orderKeys) collectExpr(ok.expr);
      break;
    case PlanNodeType.DISTINCT:
      break;
  }
}
function collectTableAliases(node) {
  const aliases = /* @__PURE__ */ new Set();
  function walk(n) {
    if (!n) return;
    if (n.type === PlanNodeType.SCAN) {
      aliases.add((n.alias || n.table).toUpperCase());
    }
    for (const child of getChildren(n)) walk(child);
  }
  walk(node);
  return aliases;
}
function hasAnyColumnUsed(tableAliases, usedColumns) {
  for (const col2 of usedColumns) {
    const table = col2.split(".")[0];
    if (tableAliases.has(table)) return true;
  }
  return false;
}
function hasAnyNameUsed(columnNames, usedColumns) {
  for (const col2 of usedColumns) {
    const name = col2.split(".")[1];
    if (name && columnNames.has(name)) return true;
  }
  return false;
}
function collectOutputNames(node) {
  const names = /* @__PURE__ */ new Set();
  const add = (value) => {
    const name = (value || "").toUpperCase();
    if (name) names.add(name);
  };
  const outputName3 = (expr2) => add(expr2?.outputName || expr2?.alias || expr2?.name || expr2?.columnName);
  function walk(n) {
    if (!n) return;
    switch (n.type) {
      case PlanNodeType.SCAN:
        for (const col2 of n.columns || []) add(col2.name || col2.columnName);
        return;
      case PlanNodeType.PROJECT:
        for (const expr2 of n.expressions || []) outputName3(expr2);
        return;
      case PlanNodeType.AGGREGATE:
        for (const expr2 of n.groupBy || []) outputName3(expr2);
        for (const agg of n.aggregates || []) outputName3(agg);
        return;
      default:
        for (const child of getChildren(n)) walk(child);
    }
  }
  walk(node);
  return names;
}

// src/optimizer/passes/predicate-dedup.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var PredicateDedup = class extends OptimizationPass {
  get name() {
    return "PredicateDedup";
  }
  apply(plan) {
    const rewriter = new DedupRewriter();
    return rewriter.rewrite(plan);
  }
};
var DedupRewriter = class extends PlanRewriter {
  rewriteFilter(node) {
    const child = this.rewrite(node.children[0]);
    const preds = splitConjuncts(node.condition);
    const unique = dedup(preds);
    if (unique.length === 0) return child;
    if (unique.length === preds.length && child === node.children[0]) return node;
    return LogicalFilter(combineConjuncts(unique), child);
  }
  rewriteJoin(node) {
    const newNode = this.rewriteChildren(node);
    if (!newNode.condition) return newNode;
    const preds = splitConjuncts(newNode.condition);
    const unique = dedup(preds);
    if (unique.length === preds.length) return newNode;
    if (unique.length === 0) return { ...newNode, condition: null };
    return { ...newNode, condition: combineConjuncts(unique) };
  }
};
function dedup(predicates) {
  const seen = /* @__PURE__ */ new Set();
  const result = [];
  for (const pred of predicates) {
    const key = exprKey3(pred);
    if (!seen.has(key)) {
      seen.add(key);
      result.push(pred);
    }
  }
  return result;
}
function exprKey3(expr2) {
  if (!expr2 || typeof expr2 !== "object") return String(expr2);
  switch (expr2.kind) {
    case BoundExprKind.COLUMN_REF:
      return `COL:${(expr2.tableAlias || "").toUpperCase()}.${(expr2.columnName || "").toUpperCase()}`;
    case BoundExprKind.LITERAL:
      return `LIT:${String(expr2.value)}:${expr2.dataType || ""}`;
    case BoundExprKind.BINARY: {
      const l = exprKey3(expr2.left);
      const r = exprKey3(expr2.right);
      const op = expr2.op;
      if (["=", "<>", "AND", "OR", "+", "*"].includes(op)) {
        return `BIN:${op}:${l < r ? l : r}:${l < r ? r : l}`;
      }
      return `BIN:${op}:${l}:${r}`;
    }
    case BoundExprKind.UNARY:
      return `UNARY:${expr2.op}:${exprKey3(expr2.operand)}`;
    case BoundExprKind.LIKE:
      return `LIKE:${expr2.negated}:${exprKey3(expr2.expr)}:${exprKey3(expr2.pattern)}`;
    case BoundExprKind.IN_LIST:
      return `IN:${expr2.negated}:${exprKey3(expr2.expr)}:${Array.isArray(expr2.list) ? expr2.list.map(exprKey3).join(",") : exprKey3(expr2.list)}`;
    case BoundExprKind.BETWEEN:
      return `BTW:${expr2.negated}:${exprKey3(expr2.expr)}:${exprKey3(expr2.low)}:${exprKey3(expr2.high)}`;
    case BoundExprKind.IS_NULL:
      return `ISNULL:${expr2.negated}:${exprKey3(expr2.expr)}`;
    case BoundExprKind.AGGREGATE:
      return `AGG:${expr2.name}:${expr2.distinct}:${expr2.args.map(exprKey3).join(",")}`;
    case BoundExprKind.FUNCTION:
      return `FN:${expr2.name}:${expr2.args.map(exprKey3).join(",")}`;
    case BoundExprKind.EXTRACT:
      return `EXT:${expr2.field}:${exprKey3(expr2.source)}`;
    case BoundExprKind.CASE:
      return `CASE:${expr2.whenClauses.map((wc) => `${exprKey3(wc.condition)}:${exprKey3(wc.result)}`).join(";")}:${exprKey3(expr2.elseExpr)}`;
    default:
      return JSON.stringify(expr2).slice(0, 100);
  }
}

// src/optimizer/passes/join-residual-split.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var JoinResidualSplit = class extends OptimizationPass {
  get name() {
    return "JoinResidualSplit";
  }
  apply(plan) {
    const rewriter = new JoinResidualSplitRewriter();
    return rewriter.rewrite(plan);
  }
};
var JoinResidualSplitRewriter = class extends PlanRewriter {
  rewriteJoin(node) {
    const rewritten = this.rewriteChildren(node);
    if (rewritten.joinType !== JoinType.INNER || !rewritten.condition) return rewritten;
    const leftRefs = collectPlanRefs3(rewritten.children[0]);
    const rightRefs = collectPlanRefs3(rewritten.children[1]);
    const joinPreds = [];
    const residualPreds = [];
    for (const pred of splitConjuncts(rewritten.condition)) {
      if (isCrossSideOr(pred, leftRefs, rightRefs)) residualPreds.push(pred);
      else joinPreds.push(pred);
    }
    if (residualPreds.length === 0 || joinPreds.length === 0) return rewritten;
    const join = LogicalJoin(
      rewritten.joinType,
      combineConjuncts(joinPreds),
      rewritten.children[0],
      rewritten.children[1],
      rewritten.physicalStrategy
    );
    return LogicalFilter(combineConjuncts(residualPreds), copyJoinMetadata(join, rewritten));
  }
};
function copyJoinMetadata(target, source) {
  const result = { ...target };
  for (const key of Object.keys(source)) {
    if (key.startsWith("_")) result[key] = source[key];
  }
  if (source.markColumn) result.markColumn = source.markColumn;
  return result;
}
function isCrossSideOr(expr2, leftRefs, rightRefs) {
  if (!expr2 || expr2.kind !== BoundExprKind.BINARY || expr2.op !== "OR") return false;
  const refs = collectExprRefs(expr2);
  return refs.some((ref) => refBelongsToPlan3(ref, leftRefs)) && refs.some((ref) => refBelongsToPlan3(ref, rightRefs));
}
function collectExprRefs(expr2) {
  const refs = [];
  walkExpr2(expr2, (e) => {
    if (e.kind === BoundExprKind.COLUMN_REF) {
      refs.push({
        tableAlias: (e.tableAlias || "").toUpperCase(),
        columnName: (e.columnName || "").toUpperCase()
      });
    }
  });
  return refs;
}
function collectPlanRefs3(node) {
  const refs = { aliases: /* @__PURE__ */ new Set(), columns: /* @__PURE__ */ new Set() };
  addOutputRefs4(node, refs);
  refs.aliases.delete("");
  refs.columns.delete("");
  return refs;
}
function addOutputRefs4(node, refs) {
  if (!node) return;
  if (node.type === PlanNodeType.SCAN) {
    refs.aliases.add((node.alias || node.table || "").toUpperCase());
    for (const col2 of node.columns || []) refs.columns.add((col2.name || col2.columnName || "").toUpperCase());
    return;
  }
  if (node.type === PlanNodeType.CTE_SCAN) {
    refs.aliases.add((node.alias || node.cteName || "").toUpperCase());
    return;
  }
  if (node.type === PlanNodeType.PROJECT) {
    for (const expr2 of node.expressions || []) refs.columns.add(outputName2(expr2));
    return;
  }
  if (node.type === PlanNodeType.AGGREGATE) {
    for (const expr2 of node.groupBy || []) refs.columns.add(outputName2(expr2));
    for (const agg of node.aggregates || []) refs.columns.add(outputName2(agg));
    return;
  }
  if (node.type === PlanNodeType.JOIN || node.type === PlanNodeType.UNION) {
    for (const child of getChildren(node)) addOutputRefs4(child, refs);
    return;
  }
  if (node.children?.[0]) addOutputRefs4(node.children[0], refs);
}
function outputName2(expr2) {
  return (expr2?.outputName || expr2?.alias || expr2?.name || expr2?.columnName || "").toUpperCase();
}
function refBelongsToPlan3(ref, planRefs) {
  if (ref.tableAlias) return planRefs.aliases.has(ref.tableAlias);
  return planRefs.columns.has(ref.columnName);
}
function walkExpr2(expr2, fn) {
  if (!expr2 || typeof expr2 !== "object") return;
  fn(expr2);
  if (expr2.left) walkExpr2(expr2.left, fn);
  if (expr2.right) walkExpr2(expr2.right, fn);
  if (expr2.operand) walkExpr2(expr2.operand, fn);
  if (expr2.expr) walkExpr2(expr2.expr, fn);
  if (expr2.low) walkExpr2(expr2.low, fn);
  if (expr2.high) walkExpr2(expr2.high, fn);
  if (expr2.args) for (const arg of expr2.args) walkExpr2(arg, fn);
  if (expr2.whenClauses) {
    for (const wc of expr2.whenClauses) {
      walkExpr2(wc.condition, fn);
      walkExpr2(wc.result, fn);
    }
  }
  if (expr2.elseExpr) walkExpr2(expr2.elseExpr, fn);
  if (expr2.list && Array.isArray(expr2.list)) for (const item of expr2.list) walkExpr2(item, fn);
  if (expr2.pattern) walkExpr2(expr2.pattern, fn);
  if (expr2.source) walkExpr2(expr2.source, fn);
}

// src/optimizer/passes/topn-fusion.js
init_buffer_shim();
init_logical_plan();
var TopNFusion = class extends OptimizationPass {
  get name() {
    return "TopNFusion";
  }
  apply(plan) {
    const rewriter = new TopNFusionRewriter();
    return rewriter.rewrite(plan);
  }
};
var TopNFusionRewriter = class extends PlanRewriter {
  rewriteLimit(node) {
    const child = this.rewrite(node.children[0]);
    if (child.type === PlanNodeType.SORT) {
      return {
        type: PlanNodeType.TOP_N,
        orderKeys: child.orderKeys,
        count: node.count,
        offset: node.offset || 0,
        children: child.children,
        _sortedBy: child._sortedBy,
        _cardinality: Math.min(node.count, child._cardinality || Infinity)
      };
    }
    if (child.type === PlanNodeType.PROJECT && child.children[0]?.type === PlanNodeType.SORT) {
      const sort = child.children[0];
      const topN = {
        type: PlanNodeType.TOP_N,
        orderKeys: sort.orderKeys,
        count: node.count,
        offset: node.offset || 0,
        children: sort.children,
        _sortedBy: sort._sortedBy,
        _cardinality: Math.min(node.count, sort._cardinality || Infinity)
      };
      return { ...child, children: [topN] };
    }
    if (child !== node.children[0]) {
      return { ...node, children: [child] };
    }
    return node;
  }
};

// src/optimizer/passes/index-selection.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
init_config();
var IndexSelection = class extends OptimizationPass {
  constructor(catalog, statistics) {
    super();
    this.catalog = catalog;
    this.statistics = statistics;
  }
  get name() {
    return "IndexSelection";
  }
  apply(plan) {
    const rewriter = new IndexSelectionRewriter(this.catalog, this.statistics);
    return rewriter.rewrite(plan);
  }
};
var IndexSelectionRewriter = class extends PlanRewriter {
  constructor(catalog, statistics) {
    super();
    this.catalog = catalog;
    this.statistics = statistics;
  }
  rewriteFilter(node) {
    const child = this.rewrite(node.children[0]);
    if (child.type !== PlanNodeType.SCAN) {
      const newNode = { ...node, children: [child] };
      return newNode;
    }
    const tableName = child.table;
    const alias = child.alias || tableName;
    const conjuncts = splitConjuncts(node.condition);
    const columnBounds = /* @__PURE__ */ new Map();
    const conjunctMapping = [];
    for (let i = 0; i < conjuncts.length; i++) {
      const info = this._analyzeConjunct(conjuncts[i], alias);
      if (!info) {
        conjunctMapping.push({ idx: i, indexed: false });
        continue;
      }
      conjunctMapping.push({ idx: i, indexed: true, column: info.column });
      if (!columnBounds.has(info.column)) {
        columnBounds.set(info.column, { point: null, low: null, high: null, lowInc: false, highInc: false, conjunctIndices: [] });
      }
      const bounds = columnBounds.get(info.column);
      bounds.conjunctIndices.push(i);
      if (info.type === "eq") {
        bounds.point = info.value;
      } else if (info.type === "gt" || info.type === "gte") {
        bounds.low = info.value;
        bounds.lowInc = info.type === "gte";
      } else if (info.type === "lt" || info.type === "lte") {
        bounds.high = info.value;
        bounds.highInc = info.type === "lte";
      }
    }
    let bestColumn = null;
    let bestBounds = null;
    for (const [col2, bounds] of columnBounds) {
      const btree = this.catalog.getIndexForColumn(tableName, col2);
      if (!btree) continue;
      if (this.statistics) {
        const tableStats = this.statistics.get(tableName.toUpperCase());
        if (tableStats) {
          const colStats = tableStats.getColumnStats(col2);
          if (colStats && colStats.ndv > 0) {
            let selectivity;
            if (bounds.point !== null) {
              selectivity = 1 / colStats.ndv;
            } else {
              selectivity = this._estimateRangeSelectivity(colStats, bounds);
            }
            if (selectivity > Config.indexScanSelectivityThreshold) continue;
          }
        }
      }
      bestColumn = col2;
      bestBounds = bounds;
      break;
    }
    if (!bestColumn) {
      return { ...node, children: [child] };
    }
    const indexedIndices = new Set(bestBounds.conjunctIndices);
    const residualConjuncts = conjuncts.filter((_, i) => !indexedIndices.has(i));
    let scanType, scanKey, scanLow, scanHigh, lowInc, highInc;
    const indexName = `idx_${tableName}_${bestColumn}`.toUpperCase();
    if (bestBounds.point !== null) {
      scanType = "point";
      scanKey = bestBounds.point;
      scanLow = null;
      scanHigh = null;
      lowInc = true;
      highInc = true;
    } else {
      scanType = "range";
      scanKey = null;
      scanLow = bestBounds.low;
      scanHigh = bestBounds.high;
      lowInc = bestBounds.lowInc;
      highInc = bestBounds.highInc;
    }
    const indexScan = LogicalIndexScan(
      tableName,
      alias,
      indexName,
      bestColumn,
      scanType,
      scanKey,
      scanLow,
      scanHigh,
      lowInc,
      highInc,
      child.columns
    );
    if (residualConjuncts.length > 0) {
      return LogicalFilter(combineConjuncts(residualConjuncts), indexScan);
    }
    return indexScan;
  }
  _estimateRangeSelectivity(colStats, bounds) {
    const min2 = toNumber3(colStats.min);
    const max2 = toNumber3(colStats.max);
    if (min2 === null || max2 === null || max2 <= min2) return 0.33;
    let low = bounds.low !== null ? toNumber3(bounds.low) : min2;
    let high = bounds.high !== null ? toNumber3(bounds.high) : max2;
    if (low === null) low = min2;
    if (high === null) high = max2;
    const covered = Math.max(0, Math.min(max2, high) - Math.max(min2, low));
    return Math.max(1e-4, covered / (max2 - min2));
  }
  _analyzeConjunct(expr2, alias) {
    if (expr2.kind !== BoundExprKind.BINARY) return null;
    const op = expr2.op;
    if (op !== "=" && op !== ">" && op !== ">=" && op !== "<" && op !== "<=") return null;
    let colExpr = null;
    let litExpr = null;
    let flipped = false;
    if (expr2.left.kind === BoundExprKind.COLUMN_REF && expr2.right.kind === BoundExprKind.LITERAL) {
      colExpr = expr2.left;
      litExpr = expr2.right;
    } else if (expr2.right.kind === BoundExprKind.COLUMN_REF && expr2.left.kind === BoundExprKind.LITERAL) {
      colExpr = expr2.right;
      litExpr = expr2.left;
      flipped = true;
    } else {
      return null;
    }
    if (colExpr.tableAlias && colExpr.tableAlias.toUpperCase() !== alias.toUpperCase()) return null;
    const column = colExpr.columnName.toUpperCase();
    const value = litExpr.value;
    let type;
    if (op === "=") {
      type = "eq";
    } else if (op === ">") {
      type = flipped ? "lt" : "gt";
    } else if (op === ">=") {
      type = flipped ? "lte" : "gte";
    } else if (op === "<") {
      type = flipped ? "gt" : "lt";
    } else if (op === "<=") {
      type = flipped ? "gte" : "lte";
    }
    return { column, value, type };
  }
};
function toNumber3(value) {
  if (value === null || value === void 0) return null;
  if (typeof value === "bigint") return Number(value);
  if (typeof value === "number") return value;
  return null;
}

// src/storage/btree.js
init_buffer_shim();
init_config();
var BTreeNode = class {
  constructor(isLeaf) {
    this.isLeaf = isLeaf;
    this.keys = [];
    this.children = [];
    this.values = [];
    this.next = null;
  }
};
var BTreeIndex = class {
  constructor(dataType) {
    this.dataType = dataType;
    this.root = new BTreeNode(true);
    this.order = Config.btreeOrder;
  }
  _compare(a, b) {
    if (a === b) return 0;
    if (a === null || a === void 0) return -1;
    if (b === null || b === void 0) return 1;
    if (typeof a === "bigint") a = Number(a);
    if (typeof b === "bigint") b = Number(b);
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
  }
  _findIndex(keys, key) {
    let lo = 0, hi = keys.length;
    while (lo < hi) {
      const mid = lo + hi >>> 1;
      if (this._compare(keys[mid], key) < 0) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
  insert(key, rowLocation) {
    const result = this._insertInternal(this.root, key, rowLocation);
    if (result) {
      const newRoot = new BTreeNode(false);
      newRoot.keys = [result.key];
      newRoot.children = [this.root, result.right];
      this.root = newRoot;
    }
  }
  _insertInternal(node, key, rowLocation) {
    if (node.isLeaf) {
      const pos2 = this._findIndex(node.keys, key);
      if (pos2 < node.keys.length && this._compare(node.keys[pos2], key) === 0) {
        node.values[pos2].push(rowLocation);
        return null;
      }
      node.keys.splice(pos2, 0, key);
      node.values.splice(pos2, 0, [rowLocation]);
      if (node.keys.length >= this.order) {
        return this._splitLeaf(node);
      }
      return null;
    }
    const childIdx = this._findChildIndex(node, key);
    const result = this._insertInternal(node.children[childIdx], key, rowLocation);
    if (!result) return null;
    const pos = this._findIndex(node.keys, result.key);
    node.keys.splice(pos, 0, result.key);
    node.children.splice(pos + 1, 0, result.right);
    if (node.keys.length >= this.order) {
      return this._splitInternal(node);
    }
    return null;
  }
  _findChildIndex(node, key) {
    let lo = 0, hi = node.keys.length;
    while (lo < hi) {
      const mid = lo + hi >>> 1;
      if (this._compare(node.keys[mid], key) <= 0) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
  _splitLeaf(leaf) {
    const mid = Math.ceil(leaf.keys.length / 2);
    const rightKeys = leaf.keys.splice(mid);
    const rightValues = leaf.values.splice(mid);
    const newRight = new BTreeNode(true);
    newRight.keys = rightKeys;
    newRight.values = rightValues;
    newRight.next = leaf.next;
    leaf.next = newRight;
    return { key: rightKeys[0], right: newRight };
  }
  _splitInternal(node) {
    const mid = Math.floor(node.keys.length / 2);
    const upKey = node.keys[mid];
    const rightKeys = node.keys.splice(mid + 1);
    node.keys.splice(mid, 1);
    const rightChildren = node.children.splice(mid + 1);
    const newRight = new BTreeNode(false);
    newRight.keys = rightKeys;
    newRight.children = rightChildren;
    return { key: upKey, right: newRight };
  }
  search(key) {
    let node = this.root;
    while (!node.isLeaf) {
      const idx = this._findChildIndex(node, key);
      node = node.children[idx];
    }
    const pos = this._findIndex(node.keys, key);
    if (pos < node.keys.length && this._compare(node.keys[pos], key) === 0) {
      return node.values[pos];
    }
    return [];
  }
  *range(low, high, lowInclusive, highInclusive) {
    let node = this.root;
    if (low !== null && low !== void 0) {
      while (!node.isLeaf) {
        const idx = this._findChildIndex(node, low);
        node = node.children[idx];
      }
    } else {
      while (!node.isLeaf) {
        node = node.children[0];
      }
    }
    let pos = 0;
    if (low !== null && low !== void 0) {
      pos = this._findIndex(node.keys, low);
      if (!lowInclusive && pos < node.keys.length && this._compare(node.keys[pos], low) === 0) {
        pos++;
      }
    }
    let current = node;
    let i = pos;
    while (current) {
      while (i < current.keys.length) {
        const k = current.keys[i];
        if (high !== null && high !== void 0) {
          const cmp = this._compare(k, high);
          if (cmp > 0 || cmp === 0 && !highInclusive) return;
        }
        for (const loc of current.values[i]) {
          yield loc;
        }
        i++;
      }
      current = current.next;
      i = 0;
    }
  }
};

// src/optimizer/passes/filter-ordering.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var FilterOrdering = class extends OptimizationPass {
  constructor(statisticsMap = /* @__PURE__ */ new Map()) {
    super();
    this.cardEstimator = new DefaultCardinalityEstimator(statisticsMap);
  }
  get name() {
    return "FilterOrdering";
  }
  apply(plan) {
    const rewriter = new FilterOrderingRewriter(this.cardEstimator);
    return rewriter.rewrite(plan);
  }
};
var FilterOrderingRewriter = class extends PlanRewriter {
  constructor(cardEstimator) {
    super();
    this.cardEstimator = cardEstimator;
  }
  rewriteFilter(node) {
    const rewritten = this.rewriteChildren(node);
    const conjuncts = splitConjuncts5(rewritten.condition);
    if (conjuncts.length < 2) return rewritten;
    const scored = conjuncts.map((pred) => ({
      pred,
      selectivity: this.cardEstimator.estimateSelectivity(pred)
    }));
    scored.sort((a, b) => a.selectivity - b.selectivity);
    const reordered = combineConjuncts4(scored.map((s) => s.pred));
    return { ...rewritten, condition: reordered };
  }
};
function splitConjuncts5(expr2) {
  if (!expr2) return [];
  if (expr2.kind === BoundExprKind.BINARY && expr2.op === "AND") {
    return [...splitConjuncts5(expr2.left), ...splitConjuncts5(expr2.right)];
  }
  return [expr2];
}
function combineConjuncts4(preds) {
  if (preds.length === 0) return null;
  if (preds.length === 1) return preds[0];
  return preds.reduce((acc, p) => ({
    kind: BoundExprKind.BINARY,
    op: "AND",
    left: acc,
    right: p,
    resultType: "BOOLEAN"
  }));
}

// src/optimizer/passes/aggregate-pushdown.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
var DECOMPOSABLE_FUNCTIONS = /* @__PURE__ */ new Map([
  ["SUM", { partial: "SUM", final: "SUM" }],
  ["COUNT", { partial: "COUNT", final: "SUM" }],
  ["MIN", { partial: "MIN", final: "MIN" }],
  ["MAX", { partial: "MAX", final: "MAX" }]
]);
var AggregatePushdown = class extends OptimizationPass {
  get name() {
    return "AggregatePushdown";
  }
  apply(plan) {
    const rewriter = new AggregatePushdownRewriter();
    return rewriter.rewrite(plan);
  }
};
var AggregatePushdownRewriter = class extends PlanRewriter {
  rewriteAggregate(node) {
    const rewritten = this.rewriteChildren(node);
    const child = rewritten.children[0];
    if (child.type !== PlanNodeType.JOIN || child.joinType !== JoinType.INNER) return rewritten;
    if (!rewritten.groupBy || rewritten.groupBy.length === 0) return rewritten;
    if (!rewritten.aggregates || rewritten.aggregates.length === 0) return rewritten;
    if (!this.allAggregatesDecomposable(rewritten.aggregates)) return rewritten;
    const leftRefs = this.collectPlanTableRefs(child.children[0]);
    const rightRefs = this.collectPlanTableRefs(child.children[1]);
    const groupBySide = this.classifyColumns(rewritten.groupBy, leftRefs, rightRefs);
    if (groupBySide === "both" || groupBySide === "none") return rewritten;
    const aggColumnSide = this.classifyAggInputs(rewritten.aggregates, leftRefs, rightRefs);
    if (aggColumnSide === "both") return rewritten;
    if (aggColumnSide !== groupBySide && aggColumnSide !== "none") return rewritten;
    const pushSide = groupBySide;
    const pushChildIdx = pushSide === "left" ? 0 : 1;
    const partialAggregates = this.buildPartialAggregates(rewritten.aggregates);
    const partialAgg = LogicalAggregate(
      [...rewritten.groupBy],
      partialAggregates,
      child.children[pushChildIdx]
    );
    const newJoinChildren = [...child.children];
    newJoinChildren[pushChildIdx] = partialAgg;
    const newJoin = { ...child, children: newJoinChildren };
    const finalAggregates = this.buildFinalAggregates(rewritten.aggregates, partialAggregates);
    return LogicalAggregate(rewritten.groupBy, finalAggregates, newJoin);
  }
  allAggregatesDecomposable(aggregates) {
    return aggregates.every((agg) => {
      if (agg.distinct) return false;
      const funcName = (agg.func || agg.functionName || "").toUpperCase();
      return DECOMPOSABLE_FUNCTIONS.has(funcName);
    });
  }
  buildPartialAggregates(aggregates) {
    return aggregates.map((agg, idx) => {
      const funcName = (agg.func || agg.functionName || "").toUpperCase();
      const rule = DECOMPOSABLE_FUNCTIONS.get(funcName);
      return {
        ...agg,
        func: rule.partial,
        functionName: rule.partial,
        outputName: agg.outputName || `_partial_${idx}`,
        _isPartial: true
      };
    });
  }
  buildFinalAggregates(originalAggs, partialAggs) {
    return originalAggs.map((agg, idx) => {
      const funcName = (agg.func || agg.functionName || "").toUpperCase();
      const rule = DECOMPOSABLE_FUNCTIONS.get(funcName);
      const partialRef = {
        kind: BoundExprKind.COLUMN_REF,
        columnName: partialAggs[idx].outputName,
        tableAlias: null
      };
      return {
        ...agg,
        func: rule.final,
        functionName: rule.final,
        args: [partialRef],
        _isFinal: true
      };
    });
  }
  classifyColumns(columns, leftRefs, rightRefs) {
    let hasLeft = false, hasRight = false;
    for (const col2 of columns) {
      if (col2.kind !== BoundExprKind.COLUMN_REF) return "both";
      const table = (col2.tableAlias || "").toUpperCase();
      if (leftRefs.has(table)) hasLeft = true;
      else if (rightRefs.has(table)) hasRight = true;
      else return "none";
    }
    if (hasLeft && hasRight) return "both";
    if (hasLeft) return "left";
    if (hasRight) return "right";
    return "none";
  }
  classifyAggInputs(aggregates, leftRefs, rightRefs) {
    let hasLeft = false, hasRight = false;
    for (const agg of aggregates) {
      const funcName = (agg.func || agg.functionName || "").toUpperCase();
      if (funcName === "COUNT" && (!agg.args || agg.args.length === 0 || this.isCountStar(agg))) continue;
      for (const arg of agg.args || []) {
        this.walkExprRefs(arg, (table) => {
          if (leftRefs.has(table)) hasLeft = true;
          else if (rightRefs.has(table)) hasRight = true;
        });
      }
    }
    if (hasLeft && hasRight) return "both";
    if (hasLeft) return "left";
    if (hasRight) return "right";
    return "none";
  }
  isCountStar(agg) {
    if (!agg.args || agg.args.length === 0) return true;
    return agg.args.length === 1 && agg.args[0]?.kind === BoundExprKind.LITERAL;
  }
  walkExprRefs(expr2, callback) {
    if (!expr2 || typeof expr2 !== "object") return;
    if (expr2.kind === BoundExprKind.COLUMN_REF && expr2.tableAlias) {
      callback(expr2.tableAlias.toUpperCase());
    }
    if (expr2.left) this.walkExprRefs(expr2.left, callback);
    if (expr2.right) this.walkExprRefs(expr2.right, callback);
    if (expr2.operand) this.walkExprRefs(expr2.operand, callback);
    if (expr2.args) for (const a of expr2.args) this.walkExprRefs(a, callback);
  }
  collectPlanTableRefs(node) {
    const refs = /* @__PURE__ */ new Set();
    this._collectRefs(node, refs);
    return refs;
  }
  _collectRefs(node, refs) {
    if (!node) return;
    if (node.type === PlanNodeType.SCAN) {
      refs.add((node.alias || node.table || "").toUpperCase());
      return;
    }
    if (node.children) {
      for (const child of node.children) this._collectRefs(child, refs);
    }
  }
};

// src/catalog/statistics-cache.js
init_buffer_shim();
var StatisticsCache = class {
  constructor(catalog) {
    this.catalog = catalog;
    this.cache = /* @__PURE__ */ new Map();
    this.versions = /* @__PURE__ */ new Map();
  }
  get(tableName) {
    const key = tableName.toUpperCase();
    const entry = this.cache.get(key);
    if (entry && entry.version === this.getVersion(key)) {
      return entry.stats;
    }
    return void 0;
  }
  has(tableName) {
    return this.get(tableName) !== void 0;
  }
  set(tableName, stats) {
    const key = tableName.toUpperCase();
    this.cache.set(key, { stats, version: this.getVersion(key) });
  }
  async ensure(tableName) {
    const key = tableName.toUpperCase();
    const existing = this.get(key);
    if (existing) return existing;
    const storage = this.catalog.getTableStorage(key);
    if (!storage) return void 0;
    const stats = await StatisticsCollector.collect(storage);
    this.set(key, stats);
    return stats;
  }
  async ensureAll() {
    for (const name of this.catalog.listTables()) {
      await this.ensure(name);
    }
  }
  invalidate(tableName) {
    const key = tableName.toUpperCase();
    this.versions.set(key, this.getVersion(key) + 1);
  }
  invalidateAll() {
    for (const key of this.cache.keys()) {
      this.invalidate(key);
    }
  }
  getVersion(key) {
    return this.versions.get(key) || 0;
  }
  get size() {
    let count2 = 0;
    for (const [key, entry] of this.cache) {
      if (entry.version === this.getVersion(key)) count2++;
    }
    return count2;
  }
  *values() {
    for (const [key, entry] of this.cache) {
      if (entry.version === this.getVersion(key)) {
        yield entry.stats;
      }
    }
  }
  *entries() {
    for (const [key, entry] of this.cache) {
      if (entry.version === this.getVersion(key)) {
        yield [key, entry.stats];
      }
    }
  }
  [Symbol.iterator]() {
    return this.entries();
  }
  toMap() {
    const map = /* @__PURE__ */ new Map();
    for (const [key, stats] of this) {
      map.set(key, stats);
    }
    return map;
  }
};

// src/engine/query-engine.js
init_logical_plan();

// src/dataframe/dataframe.js
init_buffer_shim();
init_logical_plan();
init_expression_binder();
init_data_type();

// src/dataframe/schema.js
init_buffer_shim();
var UnknownColumnError = class extends Error {
};
var AmbiguousColumnError = class extends Error {
};
var DFField = class {
  constructor(name, dataType, index, tableAlias) {
    this.name = name;
    this.dataType = dataType;
    this.index = index;
    this.tableAlias = tableAlias || "";
  }
};
function reindex(fields) {
  return fields.map((f, i) => new DFField(f.name, f.dataType, i, f.tableAlias));
}
var DFSchema = class _DFSchema {
  constructor(fields) {
    this._fields = fields;
  }
  static fromStorageSchema(storageSchema, alias) {
    return new _DFSchema(storageSchema.map((c, i) => new DFField(c.name, c.dataType, i, alias)));
  }
  static fromFields(fields) {
    return new _DFSchema(reindex(fields));
  }
  get fields() {
    return this._fields;
  }
  get length() {
    return this._fields.length;
  }
  field(i) {
    return this._fields[i];
  }
  names() {
    return this._fields.map((f) => f.name);
  }
  resolve(name, tableAlias = null) {
    const upper = name.toUpperCase();
    const aliasUpper = tableAlias ? tableAlias.toUpperCase() : null;
    let found = null;
    for (const field of this._fields) {
      if (field.name.toUpperCase() !== upper) continue;
      if (aliasUpper && field.tableAlias.toUpperCase() !== aliasUpper) continue;
      if (found) {
        throw new AmbiguousColumnError(`Ambiguous column reference: ${name}`);
      }
      found = field;
    }
    if (!found) {
      const qualifier = tableAlias ? `${tableAlias}.` : "";
      throw new UnknownColumnError(`Unknown column: ${qualifier}${name}`);
    }
    return found;
  }
  has(name, tableAlias = null) {
    try {
      this.resolve(name, tableAlias);
      return true;
    } catch (_) {
      return false;
    }
  }
  project(fields) {
    return new _DFSchema(reindex(fields.map((f) => new DFField(f.name, f.dataType, 0, f.tableAlias || ""))));
  }
  drop(names) {
    const removed = new Set(names.map((n) => n.toUpperCase()));
    const kept = this._fields.filter((f) => !removed.has(f.name.toUpperCase()));
    return new _DFSchema(reindex(kept));
  }
  append(other) {
    return new _DFSchema(reindex([...this._fields, ...other.fields]));
  }
  requalify(alias) {
    return new _DFSchema(this._fields.map((f) => new DFField(f.name, f.dataType, f.index, alias)));
  }
  static aggregateOutput(groupFields, aggFields) {
    return new _DFSchema(reindex([...groupFields, ...aggFields]));
  }
};

// src/dataframe/column-expr.js
init_buffer_shim();
init_expression_binder();
init_data_type();

// src/dataframe/type-inference.js
init_buffer_shim();
init_data_type();
var NUMERIC_RANK = {
  [DataType.INT32]: 1,
  [DataType.INT64]: 2,
  [DataType.DECIMAL]: 3,
  [DataType.FLOAT64]: 4
};
function inferValueType(value) {
  if (value === null || value === void 0) return null;
  if (typeof value === "boolean") return DataType.BOOLEAN;
  if (typeof value === "bigint") return DataType.INT64;
  if (typeof value === "number") {
    return Number.isInteger(value) ? DataType.INT32 : DataType.FLOAT64;
  }
  if (value instanceof Date) return DataType.TIMESTAMP;
  return DataType.VARCHAR;
}
function reconcileTypes(a, b) {
  if (a === null) return b;
  if (b === null) return a;
  if (a === b) return a;
  if (isNumeric(a) && isNumeric(b)) {
    return NUMERIC_RANK[a] >= NUMERIC_RANK[b] ? a : b;
  }
  return DataType.VARCHAR;
}
function inferColumnType(values) {
  let resolved = null;
  for (const value of values) {
    resolved = reconcileTypes(resolved, inferValueType(value));
  }
  return resolved === null ? DataType.VARCHAR : resolved;
}
function coerceForColumn(value, dataType) {
  if (value === null || value === void 0) return null;
  switch (dataType) {
    case DataType.BOOLEAN:
      return value;
    case DataType.INT32:
      return Number(value);
    case DataType.FLOAT64:
      return Number(value);
    case DataType.INT64:
      return BigInt(value);
    case DataType.DECIMAL:
      return BigInt(Math.round(Number(value) * DECIMAL_SCALE_NUMBER));
    case DataType.DATE:
      return value instanceof Date ? dateToEpochDays(value.getUTCFullYear(), value.getUTCMonth() + 1, value.getUTCDate()) : Number(value);
    case DataType.TIMESTAMP:
      return value instanceof Date ? BigInt(value.getTime()) : BigInt(value);
    case DataType.VARCHAR:
      return String(value);
    default:
      return value;
  }
}

// src/dataframe/expr-types.js
init_buffer_shim();
init_data_type();
function inferArithmeticType(left, right) {
  if (left === DataType.FLOAT64 || right === DataType.FLOAT64) return DataType.FLOAT64;
  if (left === DataType.DECIMAL || right === DataType.DECIMAL) return DataType.DECIMAL;
  if (left === DataType.INT64 || right === DataType.INT64) return DataType.INT64;
  if (left === DataType.DATE) return DataType.DATE;
  return DataType.INT32;
}
function inferComparisonType() {
  return DataType.BOOLEAN;
}
function inferLogicalType() {
  return DataType.BOOLEAN;
}
function inferAggregateResultType(name, argType) {
  switch (name.toUpperCase()) {
    case "COUNT":
    case "COUNT_STAR":
      return DataType.INT64;
    case "AVG":
      return DataType.FLOAT64;
    case "SUM":
    case "MIN":
    case "MAX":
      return argType || DataType.FLOAT64;
    default:
      return DataType.FLOAT64;
  }
}

// src/dataframe/sql-expr-binder.js
init_buffer_shim();
init_expression_binder();
function groupByAlias(schema) {
  const groups = /* @__PURE__ */ new Map();
  for (const field of schema.fields) {
    const key = field.tableAlias || "";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push({ name: field.name, dataType: field.dataType });
  }
  return groups;
}
function deriveOutputName(expr2) {
  if (!expr2) return null;
  if (expr2.kind === BoundExprKind.COLUMN_REF) return expr2.columnName;
  if (expr2.kind === BoundExprKind.AGGREGATE) return expr2.name.toLowerCase();
  if (expr2.kind === BoundExprKind.FUNCTION) return expr2.name.toLowerCase();
  return null;
}
function bindScalarSql(sqlString, schema, catalog, functionRegistry) {
  const ast = parseExpression(sqlString);
  const scope = new BinderScope();
  for (const [alias, columns] of groupByAlias(schema)) {
    scope.addTable(alias, { originalName: alias, columns });
  }
  const binder = new Binder(catalog, functionRegistry);
  const expr2 = binder.bindExpression(ast, scope);
  return { expr: expr2, dataType: getExprType(expr2), outputName: deriveOutputName(expr2) };
}

// src/dataframe/column-expr.js
function deriveName(expr2) {
  if (!expr2) return null;
  if (expr2.kind === BoundExprKind.COLUMN_REF) return expr2.columnName;
  if (expr2.kind === BoundExprKind.AGGREGATE) return expr2.name.toLowerCase();
  return null;
}
var Col = class _Col {
  constructor(buildFn, name = null) {
    this._build = buildFn;
    this._name = name;
    this._alias = null;
  }
  alias(name) {
    const next = new _Col(this._build, this._name);
    next._alias = name;
    return next;
  }
  as(name) {
    return this.alias(name);
  }
  bind(schema, ctx) {
    const expr2 = this._build(schema, ctx);
    const outputName3 = this._alias || this._name || deriveName(expr2);
    if (outputName3) expr2.outputName = outputName3;
    return { expr: expr2, outputName: outputName3, dataType: getExprType(expr2) };
  }
  _binary(op, other, typeFn) {
    const right = toCol(other);
    return new _Col((schema, ctx) => {
      const l = this._build(schema, ctx);
      const r = right._build(schema, ctx);
      return BoundBinary(op, l, r, typeFn(getExprType(l), getExprType(r)));
    });
  }
  add(other) {
    return this._binary("+", other, inferArithmeticType);
  }
  sub(other) {
    return this._binary("-", other, inferArithmeticType);
  }
  mul(other) {
    return this._binary("*", other, inferArithmeticType);
  }
  div(other) {
    return this._binary("/", other, inferArithmeticType);
  }
  eq(other) {
    return this._binary("=", other, inferComparisonType);
  }
  ne(other) {
    return this._binary("<>", other, inferComparisonType);
  }
  lt(other) {
    return this._binary("<", other, inferComparisonType);
  }
  le(other) {
    return this._binary("<=", other, inferComparisonType);
  }
  gt(other) {
    return this._binary(">", other, inferComparisonType);
  }
  ge(other) {
    return this._binary(">=", other, inferComparisonType);
  }
  and(other) {
    return this._binary("AND", other, inferLogicalType);
  }
  or(other) {
    return this._binary("OR", other, inferLogicalType);
  }
  not() {
    return new _Col((schema, ctx) => BoundUnary("NOT", this._build(schema, ctx), DataType.BOOLEAN));
  }
  isNull() {
    return new _Col((schema, ctx) => BoundIsNull(this._build(schema, ctx), false));
  }
  isNotNull() {
    return new _Col((schema, ctx) => BoundIsNull(this._build(schema, ctx), true));
  }
  like(pattern) {
    return new _Col((schema, ctx) => BoundLike(this._build(schema, ctx), lit(pattern)._build(schema, ctx), false));
  }
  between(low, high) {
    const lo = toCol(low);
    const hi = toCol(high);
    return new _Col((schema, ctx) => BoundBetween(this._build(schema, ctx), lo._build(schema, ctx), hi._build(schema, ctx), false));
  }
  isin(...values) {
    const cols = values.map(toCol);
    return new _Col((schema, ctx) => BoundInList(this._build(schema, ctx), cols.map((c) => c._build(schema, ctx)), false));
  }
  cast(targetType) {
    return new _Col((schema, ctx) => BoundCast(this._build(schema, ctx), targetType));
  }
};
function toCol(value) {
  return value instanceof Col ? value : lit(value);
}
function col(spec) {
  const dot = spec.indexOf(".");
  const tableAlias = dot >= 0 ? spec.slice(0, dot) : null;
  const name = dot >= 0 ? spec.slice(dot + 1) : spec;
  const built = new Col((schema) => {
    const field = schema.resolve(name, tableAlias);
    return BoundColumnRef(field.tableAlias, field.name, field.index, field.dataType);
  }, name);
  return built;
}
function lit(value) {
  return new Col(() => BoundLiteral(value, inferValueType(value)));
}
function expr(sqlString) {
  return new Col((schema, ctx) => {
    const bound = bindScalarSql(sqlString, schema, ctx.catalog, ctx.functionRegistry);
    return bound.expr;
  });
}
function aggregate(name) {
  return (column) => {
    const arg = column instanceof Col ? column : col(column);
    const built = new Col((schema, ctx) => {
      const argExpr = arg._build(schema, ctx);
      return BoundAggregate(name, [argExpr], false, inferAggregateResultType(name, getExprType(argExpr)));
    }, name.toLowerCase());
    return built;
  };
}
var sum = aggregate("SUM");
var avg = aggregate("AVG");
var min = aggregate("MIN");
var max = aggregate("MAX");
var count = aggregate("COUNT");
function countStar() {
  return new Col(() => BoundAggregate("COUNT_STAR", [], false, DataType.INT64), "count");
}

// src/dataframe/dataframe.js
var LEFT_JOIN_PREFIX = "__l";
var RIGHT_JOIN_PREFIX = "__r";
var SELF_FRAME_NAME = "self";
function fieldCol(field) {
  return new Col(
    () => BoundColumnRef(field.tableAlias, field.name, field.index, field.dataType),
    field.name
  );
}
function reconcileKeyType(left, right) {
  if (left === right) return left;
  if (left === DataType.VARCHAR) return right;
  if (right === DataType.VARCHAR) return left;
  return left;
}
function coalesceCol(leftName, rightName, dataType) {
  return new Col((schema, ctx) => BoundFunction("COALESCE", [
    col(leftName).bind(schema, ctx).expr,
    col(rightName).bind(schema, ctx).expr
  ], dataType));
}
function selectArg(item) {
  if (item instanceof Col) return item;
  return col(item);
}
function predicateArg(item) {
  if (item instanceof Col) return item;
  return expr(item);
}
function uniquify(names) {
  const seen = /* @__PURE__ */ new Map();
  return names.map((name) => {
    const key = name.toUpperCase();
    const count2 = seen.get(key) || 0;
    seen.set(key, count2 + 1);
    return count2 === 0 ? name : `${name}_${count2}`;
  });
}
var DataFrame = class _DataFrame {
  constructor(engine, plan, schema, cteMap = null) {
    this._engine = engine;
    this._plan = plan;
    this._schema = schema;
    this._cteMap = cteMap;
  }
  _ctx() {
    return { catalog: this._engine.catalog, functionRegistry: this._engine.functionRegistry };
  }
  _derive(plan, schema, extraCteMap = null) {
    return new _DataFrame(this._engine, plan, schema, mergeCteMaps(this._cteMap, extraCteMap));
  }
  columns() {
    return this._schema.names();
  }
  schema() {
    return this._schema;
  }
  explain() {
    return planToString(this._plan);
  }
  sql(sqlString) {
    const columns = this._schema.fields.map((f) => ({ name: f.name, dataType: f.dataType }));
    return this._engine.sql(sqlString, {
      frames: [{
        name: SELF_FRAME_NAME,
        columns,
        plan: this._plan,
        cteMap: this._cteMap
      }]
    });
  }
  select(...items) {
    const ctx = this._ctx();
    const bounds = items.map((item) => selectArg(item).bind(this._schema, ctx));
    const exprs = bounds.map((b) => b.expr);
    const fields = bounds.map((b, i) => new DFField(b.outputName || `col${i}`, b.dataType, i, ""));
    return this._derive(LogicalProject(exprs, this._plan), DFSchema.fromFields(fields));
  }
  filter(condition) {
    const { expr: cond, dataType } = predicateArg(condition).bind(this._schema, this._ctx());
    if (dataType !== DataType.BOOLEAN) {
      throw new TypeError(`filter condition must be boolean, got ${dataType}`);
    }
    return this._derive(LogicalFilter(cond, this._plan), this._schema);
  }
  where(condition) {
    return this.filter(condition);
  }
  withColumn(name, column) {
    const replacement = (column instanceof Col ? column : expr(column)).alias(name);
    const idx = this._schema.fields.findIndex((f) => f.name.toUpperCase() === name.toUpperCase());
    const items = idx >= 0 ? this._schema.fields.map((f, i) => i === idx ? replacement : fieldCol(f)) : [...this._schema.fields.map(fieldCol), replacement];
    return this.select(...items);
  }
  drop(...names) {
    for (const name of names) this._schema.resolve(name);
    const removed = new Set(names.map((n) => n.toUpperCase()));
    const items = this._schema.fields.filter((f) => !removed.has(f.name.toUpperCase())).map(fieldCol);
    return this.select(...items);
  }
  groupBy(...items) {
    const ctx = this._ctx();
    const bounds = items.map((item) => selectArg(item).bind(this._schema, ctx));
    return new GroupedData(this._engine, this._plan, this._schema, bounds, this._cteMap);
  }
  orderBy(...specs) {
    const ctx = this._ctx();
    const orderKeys = specs.map((spec) => {
      const descriptor = spec && typeof spec === "object" && !(spec instanceof Col) ? spec : { col: spec, desc: false };
      const { expr: keyExpr } = selectArg(descriptor.col).bind(this._schema, ctx);
      return { expr: keyExpr, direction: descriptor.desc ? SortDirection.DESC : SortDirection.ASC };
    });
    return this._derive(LogicalSort(orderKeys, this._plan), this._schema);
  }
  sort(...specs) {
    return this.orderBy(...specs);
  }
  limit(count2, offset = 0) {
    return this._derive(LogicalLimit(count2, offset, this._plan), this._schema);
  }
  distinct() {
    return this._derive(LogicalDistinct(this._plan), this._schema);
  }
  union(other) {
    return this._union(other, true);
  }
  unionAll(other) {
    return this._union(other, true);
  }
  _union(other, all) {
    if (this._schema.length !== other._schema.length) {
      throw new TypeError(`union requires equal column counts: ${this._schema.length} vs ${other._schema.length}`);
    }
    for (let i = 0; i < this._schema.length; i++) {
      const a = this._schema.field(i).dataType;
      const b = other._schema.field(i).dataType;
      if (!isComparable(a, b)) {
        throw new TypeError(`union column ${i} type mismatch: ${a} vs ${b}`);
      }
    }
    const plan = LogicalUnion(this._plan, other._plan, all);
    return this._derive(plan, DFSchema.fromFields(this._schema.fields), other._cteMap);
  }
  join(other, on, joinType = "INNER") {
    const keys = Array.isArray(on) ? on : [on];
    const type = JoinType[joinType.toUpperCase()];
    if (!type) throw new Error(`Unknown join type: ${joinType}`);
    const leftPrefix = `${LEFT_JOIN_PREFIX}${this._engine._nextDfId()}_`;
    const rightPrefix = `${RIGHT_JOIN_PREFIX}${this._engine._nextDfId()}_`;
    const leftRenamed = this.select(
      ...this._schema.fields.map((f) => fieldCol(f).alias(`${leftPrefix}${f.name}`))
    );
    const rightRenamed = other.select(
      ...other._schema.fields.map((f) => fieldCol(f).alias(`${rightPrefix}${f.name}`))
    );
    const ctx = this._ctx();
    let condition = null;
    for (const key of keys) {
      const leftField = this._schema.resolve(key);
      const rightField = other._schema.resolve(key);
      const eq = BoundBinary(
        "=",
        col(`${leftPrefix}${leftField.name}`).bind(leftRenamed._schema, ctx).expr,
        col(`${rightPrefix}${rightField.name}`).bind(rightRenamed._schema, ctx).expr,
        DataType.BOOLEAN
      );
      condition = condition ? BoundBinary("AND", condition, eq, DataType.BOOLEAN) : eq;
    }
    const preserveRight = type === JoinType.RIGHT;
    const physicalType = preserveRight ? JoinType.LEFT : type;
    const physicalChildren = preserveRight ? [rightRenamed._plan, leftRenamed._plan] : [leftRenamed._plan, rightRenamed._plan];
    const joinPlan = LogicalJoin(physicalType, condition, physicalChildren[0], physicalChildren[1]);
    const joined = new _DataFrame(
      this._engine,
      joinPlan,
      leftRenamed._schema.append(rightRenamed._schema),
      mergeCteMaps(this._cteMap, other._cteMap)
    );
    const keySet = new Set(keys.map((k) => k.toUpperCase()));
    const keyByName = new Map(keys.map((k) => {
      const lf = this._schema.resolve(k);
      const rf = other._schema.resolve(k);
      return [k.toUpperCase(), {
        left: `${leftPrefix}${lf.name}`,
        right: `${rightPrefix}${rf.name}`,
        dataType: reconcileKeyType(lf.dataType, rf.dataType)
      }];
    }));
    const projected = [];
    const outputNames = [];
    for (const field of this._schema.fields) {
      const keyInfo = keyByName.get(field.name.toUpperCase());
      projected.push(keyInfo ? coalesceCol(keyInfo.left, keyInfo.right, keyInfo.dataType) : col(`${leftPrefix}${field.name}`));
      outputNames.push(field.name);
    }
    for (const field of other._schema.fields) {
      if (keySet.has(field.name.toUpperCase())) continue;
      projected.push(col(`${rightPrefix}${field.name}`));
      outputNames.push(field.name);
    }
    const unique = uniquify(outputNames);
    return joined.select(...projected.map((item, i) => item.alias(unique[i])));
  }
  _outputColumns() {
    return this._schema.fields.map((f) => ({ name: f.name, expr: null, dataType: f.dataType }));
  }
  async collect() {
    const result = await this._engine._runPlan(this._plan, this._outputColumns(), false, this._cteMap);
    return result.toArray();
  }
  toArray() {
    return this.collect();
  }
  async count() {
    const agg = BoundAggregate("COUNT_STAR", [], false, DataType.INT64);
    agg.outputName = "count";
    const plan = LogicalAggregate([], [agg], this._plan);
    const outputColumns = [{ name: "count", expr: null, dataType: DataType.INT64 }];
    const result = await this._engine._runPlan(plan, outputColumns, false, this._cteMap);
    const rows = await result.toArray();
    return rows.length > 0 ? Number(rows[0].count) : 0;
  }
  async *chunks() {
    const result = await this._engine._runPlan(this._plan, this._outputColumns(), true, this._cteMap);
    for await (const chunk of result.chunks()) {
      yield chunk;
    }
  }
  async show(n = 20) {
    const rows = await this.limit(n).collect();
    const names = this._schema.names();
    const widths = names.map((name, i) => {
      let width = name.length;
      for (const row of rows) {
        const text = formatCell(row[names[i]]);
        if (text.length > width) width = text.length;
      }
      return width;
    });
    const renderRow = (cells) => cells.map((c, i) => c.padEnd(widths[i])).join(" | ");
    const separator = widths.map((w) => "-".repeat(w)).join("-+-");
    const lines = [renderRow(names), separator];
    for (const row of rows) {
      lines.push(renderRow(names.map((name) => formatCell(row[name]))));
    }
    const output = lines.join("\n");
    console.log(output);
    return output;
  }
};
function formatCell(value) {
  if (value === null || value === void 0) return "NULL";
  return String(value);
}
function mergeCteMaps(a, b) {
  if (!a && !b) return null;
  const merged = /* @__PURE__ */ new Map();
  if (a) for (const [k, v] of a) merged.set(k, v);
  if (b) for (const [k, v] of b) merged.set(k, v);
  return merged;
}
function aggregateGroupField(groupExpr, bound, index) {
  const name = groupExpr.kind === BoundExprKind.COLUMN_REF ? groupExpr.columnName : `group${index}`;
  const tableAlias = groupExpr.tableAlias || "";
  return {
    ref: BoundColumnRef(tableAlias, name, index, getExprType(groupExpr)),
    outputName: bound.outputName || name,
    dataType: bound.dataType
  };
}
var GroupedData = class {
  constructor(engine, childPlan, childSchema, groupBounds, cteMap) {
    this._engine = engine;
    this._childPlan = childPlan;
    this._childSchema = childSchema;
    this._groupBounds = groupBounds;
    this._cteMap = cteMap;
  }
  agg(...aggColumns) {
    const ctx = { catalog: this._engine.catalog, functionRegistry: this._engine.functionRegistry };
    const aggBounds = aggColumns.map((c) => c.bind(this._childSchema, ctx));
    const groupExprs = this._groupBounds.map((b) => b.expr);
    const aggregates = aggBounds.map((b) => b.expr);
    const aggPlan = LogicalAggregate(groupExprs, aggregates, this._childPlan);
    const groupProjections = [];
    const groupFields = [];
    for (let i = 0; i < this._groupBounds.length; i++) {
      const resolved = aggregateGroupField(groupExprs[i], this._groupBounds[i], i);
      resolved.ref.outputName = resolved.outputName;
      groupProjections.push(resolved.ref);
      groupFields.push(new DFField(resolved.outputName, resolved.dataType, i, ""));
    }
    const aggProjections = [];
    const aggFields = [];
    for (let a = 0; a < aggBounds.length; a++) {
      const bound = aggBounds[a];
      const outputName3 = bound.outputName || aggregates[a].name.toLowerCase();
      bound.expr.outputName = outputName3;
      aggProjections.push(bound.expr);
      aggFields.push(new DFField(outputName3, bound.dataType, this._groupBounds.length + a, ""));
    }
    const projectPlan = LogicalProject([...groupProjections, ...aggProjections], aggPlan);
    const schema = DFSchema.aggregateOutput(groupFields, aggFields);
    return new DataFrame(this._engine, projectPlan, schema, this._cteMap);
  }
};

// src/dataframe/in-memory-relation.js
init_buffer_shim();
init_chunk();
function buildChunks(schema, rowValues) {
  const chunks = [];
  let chunk = DataChunk.fromSchema(schema, DEFAULT_CHUNK_SIZE);
  for (const values of rowValues) {
    if (chunk.size >= DEFAULT_CHUNK_SIZE) {
      chunks.push(chunk);
      chunk = DataChunk.fromSchema(schema, DEFAULT_CHUNK_SIZE);
    }
    chunk.appendRow(values);
  }
  if (chunk.size > 0 || chunks.length === 0) chunks.push(chunk);
  return chunks;
}
var InMemoryRelation = class _InMemoryRelation {
  constructor(schema, chunks) {
    this.schema = schema;
    this.chunks = chunks;
    this._rowCount = chunks.reduce((sum2, c) => sum2 + c.size, 0);
  }
  getSchema() {
    return this.schema;
  }
  rowCount() {
    return this._rowCount;
  }
  getColumnIndex(name) {
    const upper = name.toUpperCase();
    return this.schema.findIndex((c) => c.name.toUpperCase() === upper);
  }
  async *scan() {
    for (const chunk of this.chunks) {
      yield chunk;
    }
  }
  static fromRows(rows, declaredSchema = null) {
    const names = declaredSchema ? declaredSchema.map((c) => c.name) : rows.length > 0 ? Object.keys(rows[0]) : [];
    const extract = Array.isArray(rows[0]) ? (row, i) => row[i] : (row, i) => row[names[i]];
    const columnValues = names.map((_, i) => rows.map((row) => extract(row, i)));
    const schema = names.map((name, i) => ({
      name,
      dataType: declaredSchema ? declaredSchema[i].dataType : inferColumnType(columnValues[i])
    }));
    const rowValues = rows.map((row) => schema.map((col2, i) => coerceForColumn(extract(row, i), col2.dataType)));
    return new _InMemoryRelation(schema, buildChunks(schema, rowValues));
  }
  static fromColumns(columns, declaredSchema = null) {
    const names = Object.keys(columns);
    const length = names.length > 0 ? columns[names[0]].length : 0;
    const declaredByName = new Map((declaredSchema || []).map((c) => [c.name, c.dataType]));
    const schema = names.map((name) => ({
      name,
      dataType: declaredByName.has(name) ? declaredByName.get(name) : inferColumnType(columns[name])
    }));
    const rowValues = [];
    for (let r = 0; r < length; r++) {
      rowValues.push(schema.map((col2) => coerceForColumn(columns[col2.name][r], col2.dataType)));
    }
    return new _InMemoryRelation(schema, buildChunks(schema, rowValues));
  }
};

// src/engine/query-engine.js
var DATAFRAME_TABLE_PREFIX = "__DF";
var _defaultBackendFactory = null;
function setDefaultStorageBackend(factory) {
  _defaultBackendFactory = factory;
}
var QueryEngine = class {
  constructor(catalog, options = {}) {
    this.catalog = catalog;
    this.functionRegistry = defaultFunctionRegistry;
    const backend = options.storageBackend ?? _defaultBackendFactory?.(options);
    if (!backend) {
      throw new Error("QueryEngine requires a storageBackend (none provided and no default registered)");
    }
    this.storageBackend = backend;
    this.tempManager = backend.createTempSpace();
    this.executor = new QueryExecutor(catalog, this.tempManager, backend);
    this.wasmEnabled = false;
    this._dfIdCounter = 0;
    this.precomputedStats = options.statistics || null;
    this.statsCache = new StatisticsCache(catalog);
    this.optimizer = this.createOptimizer(this.precomputedStats);
  }
  _nextDfId() {
    return this._dfIdCounter++;
  }
  close() {
    this.tempManager.cleanup();
  }
  async collectStatistics() {
    await this.statsCache.ensureAll();
    const map = this.statsCache.toMap();
    return map.size > 0 ? map : null;
  }
  createOptimizer(statistics) {
    const statsMap = statistics || /* @__PURE__ */ new Map();
    const optimizer = new Optimizer();
    optimizer.registerPass(new ExpressionSimplifier());
    optimizer.registerPass(new SubqueryUnnesting());
    optimizer.registerPass(new HavingPushdown());
    optimizer.registerPass(new CTEOptimization());
    optimizer.registerPass(new PredicatePushdown());
    optimizer.registerPass(new PredicateInference());
    optimizer.registerPass(new PredicatePushdown());
    optimizer.registerPass(new OuterToInnerJoin());
    optimizer.registerPass(new PredicatePushdown());
    optimizer.registerPass(new AggregatePushdown());
    if (statistics) {
      optimizer.registerPass(new JoinReorder(statistics));
      optimizer.registerPass(new PredicatePushdown());
    }
    optimizer.registerPass(new JoinElimination());
    optimizer.registerPass(new ProjectionPushdown());
    optimizer.registerPass(new LimitPushdown());
    optimizer.registerPass(new EmptyPropagation());
    optimizer.registerPass(new NodeMerge());
    optimizer.registerPass(new PredicateDedup());
    optimizer.registerPass(new FilterOrdering(statsMap));
    optimizer.registerPass(new IndexSelection(this.catalog, statistics));
    optimizer.registerPass(new JoinResidualSplit());
    optimizer.registerPass(new PhysicalDesign(statsMap));
    optimizer.registerPass(new SortElimination());
    optimizer.registerPass(new TopNFusion());
    return optimizer;
  }
  parseSQL(sql) {
    return parse(sql);
  }
  bind(ast) {
    const binder = new Binder(this.catalog, this.functionRegistry);
    return binder.bind(ast);
  }
  plan(boundQuery) {
    return createLogicalPlan(boundQuery);
  }
  optimize(logicalPlan) {
    return this.optimizer.optimize(logicalPlan);
  }
  async compile(sql) {
    const ast = this.parseSQL(sql);
    let isExplain = false;
    let isAnalyze = false;
    let targetAst = ast;
    if (ast.kind === "ExplainStmt") {
      isExplain = true;
      targetAst = ast.query;
    } else if (ast.kind === "ExplainAnalyzeStmt") {
      isExplain = true;
      isAnalyze = true;
      targetAst = ast.query;
    } else if (ast.kind === "CreateTableStmt" || ast.kind === "DropTableStmt") {
      return { ddl: ast };
    }
    const bound = this.bind(targetAst);
    const logicalPlan = this.plan(bound);
    let cteMap = logicalPlan._cteMap || /* @__PURE__ */ new Map();
    await this._ensureStatistics();
    const optimized = this.optimize(logicalPlan);
    cteMap = this.optimizeCTEMap(cteMap);
    return { plan: optimized, outputColumns: bound.outputColumns, cteMap, isExplain, isAnalyze };
  }
  optimizeCTEMap(cteMap) {
    if (!cteMap || cteMap.size === 0) return cteMap;
    const optimized = /* @__PURE__ */ new Map();
    for (const [name, plan] of cteMap) {
      optimized.set(name, this.optimize(plan));
    }
    return optimized;
  }
  async _ensureStatistics() {
    if (!this.precomputedStats && !this._statsCollected) {
      const collected = await this.collectStatistics();
      if (collected) {
        this.optimizer = this.createOptimizer(collected);
        this._statsCollected = true;
        if (this._distributedPasses) {
          for (const { method, args } of this._distributedPasses) {
            this.optimizer[method](...args);
          }
        }
      }
    }
  }
  table(name) {
    const tableDef = this.catalog.getTable(name);
    if (!tableDef) throw new Error(`Unknown table: ${name}`);
    const plan = LogicalScan(tableDef.name, tableDef.columns, tableDef.name);
    const schema = DFSchema.fromStorageSchema(tableDef.columns, tableDef.name);
    return new DataFrame(this, plan, schema);
  }
  createDataFrame(rows, declaredSchema = null) {
    const relation = InMemoryRelation.fromRows(rows, declaredSchema);
    const name = `${DATAFRAME_TABLE_PREFIX}${this._nextDfId()}`;
    const storageSchema = relation.getSchema();
    this.catalog.registerTable(name, storageSchema);
    this.catalog.registerTableStorage(name, relation);
    const plan = LogicalScan(name, storageSchema, name);
    const schema = DFSchema.fromStorageSchema(storageSchema, name);
    return new DataFrame(this, plan, schema);
  }
  sql(sqlString, options = {}) {
    const ast = this.parseSQL(sqlString);
    if (ast.kind === "CreateTableStmt" || ast.kind === "DropTableStmt" || ast.kind === "ExplainStmt" || ast.kind === "ExplainAnalyzeStmt") {
      throw new Error("sql() supports query statements only");
    }
    const binder = new Binder(this.catalog, this.functionRegistry);
    for (const frame of options.frames || []) {
      binder.cteScopes.set(frame.name.toUpperCase(), {
        name: frame.name,
        columns: frame.columns,
        bound: { prebuiltPlan: frame.plan }
      });
    }
    const bound = binder.bind(ast);
    const logicalPlan = this.plan(bound);
    let cteMap = logicalPlan._cteMap || null;
    for (const frame of options.frames || []) {
      if (!frame.cteMap) continue;
      cteMap = cteMap || /* @__PURE__ */ new Map();
      for (const [k, v] of frame.cteMap) if (!cteMap.has(k)) cteMap.set(k, v);
    }
    const schema = DFSchema.fromFields(bound.outputColumns.map((c, i) => ({
      name: c.name,
      dataType: c.dataType,
      index: i,
      tableAlias: ""
    })));
    return new DataFrame(this, logicalPlan, schema, cteMap);
  }
  async _runPlan(plan, outputColumns, streaming = false, cteMap = null) {
    await this._ensureStatistics();
    const optimized = this.optimize(plan);
    this.executor.cteDefinitions = this.optimizeCTEMap(cteMap || /* @__PURE__ */ new Map());
    const { sink, columnNames } = await this.executor.execute(optimized, outputColumns, streaming);
    return new QueryResult(columnNames, sink);
  }
  async run(sql) {
    const compiled = await this.compile(sql);
    if (compiled.ddl) {
      return this.executeDDL(compiled.ddl);
    }
    const { plan, outputColumns, cteMap, isExplain, isAnalyze } = compiled;
    if (isExplain && !isAnalyze) {
      const { formatPlan: formatPlan2 } = await Promise.resolve().then(() => (init_plan_formatter(), plan_formatter_exports));
      const planStr = formatPlan2(plan);
      return { rows: [{ "EXPLAIN_PLAN": planStr }], columns: ["EXPLAIN_PLAN"] };
    }
    if (isAnalyze) {
      const { formatPlan: formatPlan2 } = await Promise.resolve().then(() => (init_plan_formatter(), plan_formatter_exports));
      const planStr = formatPlan2(plan);
      const startTime = performance.now();
      this.executor.cteDefinitions = cteMap;
      const { sink, columnNames } = await this.executor.execute(plan, outputColumns);
      const result = new QueryResult(columnNames, sink);
      const rows = await result.toArray();
      const elapsed = (performance.now() - startTime).toFixed(2);
      const analyzeStr = `${planStr}
Execution Time: ${elapsed} ms
Rows Returned: ${rows.length}`;
      return { rows: [{ "EXPLAIN_ANALYZE": analyzeStr }], columns: ["EXPLAIN_ANALYZE"] };
    }
    this._activeCancel = new AbortController();
    try {
      this.executor.cteDefinitions = cteMap;
      const { sink, columnNames } = await this.executor.execute(plan, outputColumns);
      const result = new QueryResult(columnNames, sink);
      return { rows: await result.toArray(), columns: columnNames };
    } finally {
      this._activeCancel = null;
    }
  }
  cancel() {
    if (this._activeCancel) {
      this._activeCancel.abort();
    }
  }
  executeDDL(ddl) {
    if (ddl.kind === "CreateTableStmt") {
      return this.executeCreateTable(ddl);
    }
    if (ddl.kind === "DropTableStmt") {
      return this.executeDropTable(ddl);
    }
    throw new Error(`Unknown DDL: ${ddl.kind}`);
  }
  executeCreateTable(stmt) {
    const resolveType = (typeName) => {
      const map = {
        "INTEGER": DataType.INT32,
        "INT": DataType.INT32,
        "INT32": DataType.INT32,
        "BIGINT": DataType.INT64,
        "INT64": DataType.INT64,
        "FLOAT": DataType.FLOAT64,
        "DOUBLE": DataType.FLOAT64,
        "REAL": DataType.FLOAT64,
        "DECIMAL": DataType.DECIMAL,
        "NUMERIC": DataType.DECIMAL,
        "VARCHAR": DataType.VARCHAR,
        "TEXT": DataType.VARCHAR,
        "CHAR": DataType.VARCHAR,
        "DATE": DataType.DATE,
        "TIMESTAMP": DataType.TIMESTAMP,
        "DATETIME": DataType.TIMESTAMP,
        "BOOLEAN": DataType.BOOLEAN,
        "BOOL": DataType.BOOLEAN
      };
      return map[typeName.name.toUpperCase()] || DataType.VARCHAR;
    };
    const tableName = stmt.name.toUpperCase();
    if (this.catalog.getTable(tableName)) {
      if (stmt.ifNotExists) {
        return { rows: [], columns: [], message: `Table ${tableName} already exists` };
      }
      throw new Error(`Table ${tableName} already exists`);
    }
    const columns = stmt.columns.map((col2) => ({
      name: col2.name.toUpperCase(),
      dataType: resolveType(col2.typeName)
    }));
    const pageStore = this.storageBackend.createPageStore(this.tempManager.allocate("buffer", tableName));
    const table = new Table(tableName, columns, pageStore);
    this.catalog.registerTable(tableName, columns);
    this.catalog.registerTableStorage(tableName, table);
    return { rows: [], columns: [], message: `Table ${tableName} created` };
  }
  executeDropTable(stmt) {
    const tableName = stmt.name.toUpperCase();
    if (!this.catalog.getTable(tableName)) {
      if (stmt.ifExists) {
        return { rows: [], columns: [], message: `Table ${tableName} does not exist` };
      }
      throw new Error(`Table ${tableName} does not exist`);
    }
    this.catalog.dropTable(tableName);
    return { rows: [], columns: [], message: `Table ${tableName} dropped` };
  }
  async stream(sql) {
    const compiled = await this.compile(sql);
    if (compiled.ddl) return this.executeDDL(compiled.ddl);
    const { plan, outputColumns, cteMap, isExplain, isAnalyze } = compiled;
    if (isExplain && !isAnalyze) {
      const { formatPlan: formatPlan2 } = await Promise.resolve().then(() => (init_plan_formatter(), plan_formatter_exports));
      const planStr = formatPlan2(plan);
      return { rows: [{ "EXPLAIN_PLAN": planStr }], columns: ["EXPLAIN_PLAN"] };
    }
    this.executor.cteDefinitions = cteMap;
    const { sink, columnNames } = await this.executor.execute(plan, outputColumns, true);
    return new QueryResult(columnNames, sink);
  }
  async buildIndexes() {
    for (const tableName of this.catalog.listTables()) {
      const tableDef = this.catalog.getTable(tableName);
      if (!tableDef.primaryKey || tableDef.primaryKey.length === 0) continue;
      const storage = this.catalog.getTableStorage(tableName);
      if (!storage) continue;
      for (const pkCol of tableDef.primaryKey) {
        const colIdx = tableDef.columns.findIndex((c) => c.name.toUpperCase() === pkCol.toUpperCase());
        if (colIdx < 0) continue;
        const colDef = tableDef.columns[colIdx];
        const btree = new BTreeIndex(colDef.dataType);
        await storage.flush();
        for (let p = 0; p < storage.pageIds.length; p++) {
          const pageId = storage.pageIds[p];
          const chunk = await storage.bufferPool.fetchPage(pageId, true);
          for (let r = 0; r < chunk.size; r++) {
            const key = chunk.columns[colIdx].get(r);
            if (key !== null && key !== void 0) {
              btree.insert(key, { pageId, rowIndex: r });
            }
          }
        }
        this.catalog.registerIndex(tableName, pkCol, btree);
        storage.registerIndex(colIdx, btree);
      }
    }
  }
  async enableWasm() {
    try {
      const { getGlobalLoader: getGlobalLoader2 } = await Promise.resolve().then(() => (init_loader(), loader_exports));
      const loader = await getGlobalLoader2();
      await loader.loadModule("core");
      const { registerAllKernels: registerAllKernels2 } = await Promise.resolve().then(() => (init_register_kernels(), register_kernels_exports));
      registerAllKernels2();
      this.wasmEnabled = true;
    } catch (_) {
      this.wasmEnabled = false;
    }
  }
  async enableParallel() {
    const { Config: Config2 } = await Promise.resolve().then(() => (init_config(), config_exports));
    if (Config2.parallelWorkers <= 1) return false;
    try {
      const { getGlobalLoader: getGlobalLoader2 } = await Promise.resolve().then(() => (init_loader(), loader_exports));
      const loader = await getGlobalLoader2({ shared: true });
      const instance = await loader.loadModule("core");
      const regionAllocator = loader.initRegions(Config2.regionSize);
      const { registerAllKernels: registerAllKernels2 } = await Promise.resolve().then(() => (init_register_kernels(), register_kernels_exports));
      registerAllKernels2();
      this.wasmEnabled = true;
      const wasmModule = loader.getModule("core");
      const { WorkerPool } = await import("../parallel/worker-pool.js");
      const pool = new WorkerPool({
        maxWorkers: Config2.parallelWorkers,
        wasmModule,
        wasmMemory: loader.memory,
        regionAllocator
      });
      await pool.init();
      const { globalDispatch: globalDispatch2 } = await Promise.resolve().then(() => (init_dispatch(), dispatch_exports));
      const { ParallelDispatch } = await import("../parallel/parallel-dispatch.js");
      const parallelDispatch = new ParallelDispatch(pool, regionAllocator, globalDispatch2);
      const { FragmentPool } = await import("../parallel/fragment-pool.js");
      const fragmentPool = new FragmentPool(Config2.parallelWorkers, Config2.aggMorselRows);
      this.executor.setParallelContext(pool, parallelDispatch, fragmentPool);
      this.workerPool = pool;
      this.fragmentPool = fragmentPool;
      this.parallelEnabled = true;
      return true;
    } catch (_) {
      this.parallelEnabled = false;
      return false;
    }
  }
  async enableDistributed(clusterConfig = {}) {
    const { NodeDescriptor, NodeRole } = await import("../distributed/cluster/node-descriptor.js");
    const { ClusterManager } = await import("../distributed/cluster/cluster-manager.js");
    const { PartitionMap } = await import("../distributed/partition/partition-map.js");
    const { DistributionAwareJoin } = await import("../distributed/optimizer/distribution-aware-join.js");
    const { PartialAggregatePass } = await import("../distributed/optimizer/partial-aggregate.js");
    const { DistributedSortPass } = await import("../distributed/optimizer/distributed-sort.js");
    const { QueryCoordinator } = await import("../distributed/execution/coordinator.js");
    const localNode = new NodeDescriptor({
      nodeId: clusterConfig.nodeId || `node-${Date.now()}`,
      host: clusterConfig.host || "127.0.0.1",
      port: clusterConfig.port || 9400,
      role: clusterConfig.role || NodeRole.HYBRID,
      capacity: clusterConfig.capacity
    });
    const clusterManager = new ClusterManager(localNode);
    const partitionMap = new PartitionMap();
    const statsMap = this.precomputedStats || /* @__PURE__ */ new Map();
    this.optimizer.insertPassAfter("PhysicalDesign", new DistributionAwareJoin(partitionMap, statsMap));
    this.optimizer.registerPass(new PartialAggregatePass());
    this.optimizer.registerPass(new DistributedSortPass());
    this._distributedPasses = [
      { method: "insertPassAfter", args: ["PhysicalDesign", new DistributionAwareJoin(partitionMap, statsMap)] },
      { method: "registerPass", args: [new PartialAggregatePass()] },
      { method: "registerPass", args: [new DistributedSortPass()] }
    ];
    let transport = clusterConfig.transport || null;
    if (!transport) {
      const { HttpTransport } = await import("../distributed/transport/http-transport.js");
      transport = new HttpTransport({ port: localNode.port });
    }
    const coordinator = new QueryCoordinator(this, clusterManager, partitionMap, transport);
    this.distributed = {
      clusterManager,
      partitionMap,
      transport,
      coordinator,
      localNode
    };
    return coordinator;
  }
  async shutdown() {
    if (this.workerPool) {
      await this.workerPool.shutdown();
      this.workerPool = null;
    }
    if (this.fragmentPool) {
      await this.fragmentPool.close();
      this.fragmentPool = null;
    }
    if (this.distributed?.transport) {
      await this.distributed.transport.stop();
    }
    this.tempManager.cleanup();
  }
};

// src/browser.js
init_loader();

// src/wasm/fetch-byte-source.js
init_buffer_shim();
var WASM_EXTENSION = ".wasm";
var _baseUrl = null;
function setWasmBaseUrl(url) {
  _baseUrl = url;
}
function resolveWasmUrl(name) {
  const fileName = `${name}${WASM_EXTENSION}`;
  if (_baseUrl) return new URL(fileName, _baseUrl).href;
  return new URL(fileName, import.meta.url).href;
}
async function fetchByteSource(name) {
  const response = await fetch(resolveWasmUrl(name));
  if (!response.ok) {
    throw new Error(`Failed to fetch wasm module '${name}': ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

// src/catalog/catalog.js
init_buffer_shim();
var Catalog = class {
  constructor() {
    this.tables = /* @__PURE__ */ new Map();
    this.tableStorage = /* @__PURE__ */ new Map();
    this.indexes = /* @__PURE__ */ new Map();
    this.partitionInfo = /* @__PURE__ */ new Map();
  }
  registerTable(name, schema, options = {}) {
    const upperName = name.toUpperCase();
    this.tables.set(upperName, {
      name: upperName,
      columns: schema,
      primaryKey: options.primaryKey || [],
      foreignKeys: options.foreignKeys || []
    });
  }
  registerTableStorage(name, storage) {
    this.tableStorage.set(name.toUpperCase(), storage);
  }
  getTable(name) {
    return this.tables.get(name.toUpperCase()) || null;
  }
  getTableStorage(name) {
    return this.tableStorage.get(name.toUpperCase()) || null;
  }
  getColumn(tableName, columnName) {
    const table = this.getTable(tableName);
    if (!table) return null;
    const upper = columnName.toUpperCase();
    return table.columns.find((c) => c.name.toUpperCase() === upper) || null;
  }
  getColumnIndex(tableName, columnName) {
    const table = this.getTable(tableName);
    if (!table) return -1;
    const upper = columnName.toUpperCase();
    return table.columns.findIndex((c) => c.name.toUpperCase() === upper);
  }
  dropTable(name) {
    const upper = name.toUpperCase();
    this.tables.delete(upper);
    this.tableStorage.delete(upper);
    for (const [key, entry] of this.indexes) {
      if (entry.tableName === upper) this.indexes.delete(key);
    }
  }
  hasTable(name) {
    return this.tables.has(name.toUpperCase());
  }
  listTables() {
    return Array.from(this.tables.keys());
  }
  registerIndex(tableName, columnName, btree) {
    const key = `${tableName.toUpperCase()}.${columnName.toUpperCase()}`;
    this.indexes.set(key, { tableName: tableName.toUpperCase(), columnName: columnName.toUpperCase(), btree });
  }
  getIndexForColumn(tableName, columnName) {
    const key = `${tableName.toUpperCase()}.${columnName.toUpperCase()}`;
    const entry = this.indexes.get(key);
    return entry ? entry.btree : null;
  }
  getIndexesForTable(tableName) {
    const upper = tableName.toUpperCase();
    const result = [];
    for (const entry of this.indexes.values()) {
      if (entry.tableName === upper) result.push(entry);
    }
    return result;
  }
  registerPartitionInfo(tableName, strategy, partitionCount, partitionKey) {
    this.partitionInfo.set(tableName.toUpperCase(), { strategy, partitionCount, partitionKey });
  }
  getPartitionInfo(tableName) {
    return this.partitionInfo.get(tableName.toUpperCase()) || null;
  }
};

// src/browser.js
init_data_type();

// src/dataframe/index.js
init_buffer_shim();

// src/browser.js
setDefaultStorageBackend((options) => new MemoryStorageBackend(options));
configureWasmSource(fetchByteSource);
function createEngine(options = {}) {
  const catalog = options.catalog || new Catalog();
  return new QueryEngine(catalog, options);
}
function registerTable(engine, name, rows, declaredSchema = null) {
  const relation = InMemoryRelation.fromRows(rows, declaredSchema);
  const schema = relation.getSchema();
  engine.catalog.registerTable(name, schema);
  engine.catalog.registerTableStorage(name, relation);
  return schema;
}
export {
  Catalog,
  Col,
  DataFrame,
  DataType,
  GroupedData,
  InMemoryRelation,
  QueryEngine,
  avg,
  col,
  count,
  countStar,
  createEngine,
  expr,
  lit,
  max,
  min,
  registerTable,
  setWasmBaseUrl,
  sum
};
//# sourceMappingURL=query-engine.browser.js.map
