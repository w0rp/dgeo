module dgeo.point;

import core.simd;
static import std.math;

// This flag can be changed for testing the code without SIMD.
private enum disableSIMD = false;

// Credit to the exact versions to check goes to Manu Evans for the std.simd
// which never arrived.
static if (disableSIMD) {
    version = NoSIMD;
} else {
    version(X86) {
        version(DigitalMars) {
            // DMD-x86 does not support SIMD
            version = NoSIMD;
        } else {
            version = SIMD_X86_OR_X64;
        }
    } else version(X86_64) {
        version = SIMD_X86_OR_X64;
    } else {
        // Disable SIMD for architectures we haven't tested.
        version = NoSIMD;
    }
}

version(NoSIMD) {
    /**
     * Construct a scalar value for use in point or vector calculations in
     * an efficient manner, such that it will work with or without SIMD
     * support.
     */
    enum float Scalar(float value) = value;
} else {
    version = SIMD;

    /**
     * Construct a scalar value for use in point or vector calculations in
     * an efficient manner, such that it will work with or without SIMD
     * support.
     */
    enum float4 Scalar(float value) = [value, value, value, 1];

    @safe pure nothrow @nogc
    private float4 shuffle(ubyte mask)(float4 value) {
        return __simd(XMM.SHUFPS, value, value, mask);
    }

    /// Load a float into all elements of a float4 vector.
    @safe pure nothrow @nogc
    private float4 loadScalar(float value) {
        // Load a scalar into the first element.
        auto vec = __simd(XMM.LODSS, value);

        // Shuffle everything around so the scalar ends up in all elements.
        return shuffle!0b00_00_00_00(vec);
    }
}

private mixin template CommonDataAndTypes(T) {
    version(SIMD_X86_OR_X64) {
        package union {
            // Set the w value in the vector always appropriately
            // for homogenous co-ordinates.
            static if(is(T == Point)) {
                float4 xyzw = [0, 0, 0, 0];
            } else {
                float4 xyzw = [0, 0, 0, 1];
            }

            float[4] xyzwArray;
        }

        @safe pure nothrow @nogc
        @property float x() const {
            return xyzwArray[0];
        }

        @safe pure nothrow @nogc
        @property void x(float value) {
            xyzwArray[0] = value;
        }

        @safe pure nothrow @nogc
        @property float y() const {
            return xyzwArray[1];
        }

        @safe pure nothrow @nogc
        @property void y(float value) {
            xyzwArray[1] = value;
        }

        @safe pure nothrow @nogc
        @property float z() const {
            return xyzwArray[2];
        }

        @safe pure nothrow @nogc
        @property void z(float value) {
            xyzwArray[2] = value;
        }

        // Reset the w co-ordinate. This should be called after setting
        // the vector value directly to keep the w co-ordinate normalised
        // so the vector can be used in homogeneous co-ordinates.
        @safe pure nothrow @nogc
        package void resetW() {
            static if(is(T == Point)) {
                this.xyzwArray[3] = 0f;
            } else {
                this.xyzwArray[3] = 1f;
            }
        }

        @safe pure nothrow @nogc
        package this(float4 xyzw) {
            this.xyzw = xyzw;
            this.resetW();
        }
    } else {
        float x = 0f;
        float y = 0f;
        float z = 0f;

        // To keep the padding for the struct always the same, add in
        // a fourth float value for homogeneous coordinates.
        static if(is(T == Point)) {
            private immutable float w = 0f;
        } else {
            private immutable float w = 1f;
        }
    }

    /// Construct a point or vector with some co-ordinates.
    @safe pure nothrow @nogc
    this(float x, float y, float z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    /// Test if two points or two vectors are equal.
    @trusted pure nothrow @nogc
    bool opEquals(in T other) const {
        version(SIMD_X86_OR_X64) {
            enum EQ = 0;
            enum ulong[2] allOnes = [ulong.max, ulong.max];

            // Execute the SIMD equality instruction.
            auto vecResult = __simd(XMM.CMPPS, this.xyzw, other.xyzw, EQ);

            // If the two values are equal, then all bits of the result
            // will be set to 1.
            return *(cast(ulong[2]*)(&vecResult)) == allOnes;
        } else {
            return this.x == other.x
                && this.y == other.y
                && this.z == other.z;
        }
    }
}

/**
 * This type represents a point in Cartesian co-ordinates.
 *
 * All co-ordinates in a point will be default-initialised to 0.
 */
struct Point {
    mixin CommonDataAndTypes!Point;

    /// Add a vector to a point
    @safe pure nothrow @nogc
    Point opBinary(string s)(in Vector other) const if (s == "+") {
        version(SIMD_X86_OR_X64) {
            return Point(this.xyzw + other.xyzw);
        } else {
            return Point(
                this.x + other.x,
                this.y + other.y,
                this.z + other.z
            );
        }
    }

    /// Subtract two points and yield a vector.
    @safe pure nothrow @nogc
    Vector opBinary(string s)(in Point other) const if (s == "-") {
        version(SIMD_X86_OR_X64) {
            return Vector(this.xyzw - other.xyzw);
        } else {
            return Vector(
                this.x - other.x,
                this.y - other.y,
                this.z - other.z
            );
        }
    }
}

/**
 * This type represents a vector in Cartesian co-ordinates.
 *
 * All co-ordinates in a vector will be default-initialised to 0.
 */
struct Vector {
    mixin CommonDataAndTypes!Vector;

    /// Negate a vector.
    @safe pure nothrow @nogc
    Vector opUnary(string s)() const if (s == "-") {
        version(SIMD_X86_OR_X64) {
            return Vector(-this.xyzw);
        } else {
            return Vector(-this.x, -this.y, -this.z);
        }
    }

    /// Add a vector to a vector
    @safe pure nothrow @nogc
    Vector opBinary(string s)(in Vector other) const if (s == "+") {
        version(SIMD_X86_OR_X64) {
            return Vector(this.xyzw + other.xyzw);
        } else {
            return Vector(
                this.x + other.x,
                this.y + other.y,
                this.z + other.z
            );
        }
    }

    /// Subtract two vectors and yield a vector.
    @safe pure nothrow @nogc
    Vector opBinary(string s)(in Vector other) const if (s == "-") {
        version(SIMD_X86_OR_X64) {
            return Vector(this.xyzw - other.xyzw);
        } else {
            return Vector(
                this.x - other.x,
                this.y - other.y,
                this.z - other.z
            );
        }
    }

    /**
     * Multiply a vector by a scalar.
     *
     * Params:
     * scalar = A scalar multiplier.
     *
     * Returns: A new vector offset by the scalar.
     */
    @safe pure nothrow @nogc
    Vector opBinary(string s)(in float scalar) const if (s == "*") {
        version(SIMD_X86_OR_X64) {
            return Vector(cast(float4) this.xyzw * loadScalar(scalar));
        } else {
            return Vector(
                this.x * scalar,
                this.y * scalar,
                this.z * scalar
            );
        }
    }

    /// ditto
    version(SIMD)
    @safe pure nothrow @nogc
    Vector opBinary(string s)(in float4 scalar) const if (s == "*") {
        return Vector(cast(float4) this.xyzw * cast(float4) scalar);
    }

    /// ditto
    @safe pure nothrow @nogc
    Vector opBinaryRight(string s)(in float scalar) const if (s == "*") {
        return this * scalar;
    }

    /// ditto
    version(SIMD)
    @safe pure nothrow @nogc
    Vector opBinaryRight(string s)(in float4 scalar) const if (s == "*") {
        return this * scalar;
    }

    /**
     * Compute the dot product of two vectors.
     *
     * Params:
     * other = The other vector.
     *
     * Returns: The result of the dot product.
     */
    @safe pure nothrow @nogc
    float opBinary(string s)(in Vector other) const if (s == "*") {
        version(SIMD_X86_OR_X64) {
            float4 multResult = this.xyzw * other.xyzw;

            // Shuffle and add to get x + y + z.
            float4 result = multResult
                + shuffle!1(multResult)
                + shuffle!2(multResult);

            return result.array[0];
        } else {
            return this.x * other.x + this.y * other.y + this.z * other.z;
        }
    }

    /**
     * Divide a vector by a scalar.
     *
     * Params:
     * scalar = A scalar divisor.
     *
     * Returns: A new vector, with each dimension divided by the scalar.
     */
    @safe pure nothrow @nogc
    Vector opBinary(string s)(in float scalar) const if (s == "/") {
        version(SIMD_X86_OR_X64) {
            return Vector(cast(float4) this.xyzw / loadScalar(scalar));
        } else {
            return Vector(
                this.x / scalar,
                this.y / scalar,
                this.z / scalar
            );
        }
    }

    /// ditto
    version(SIMD)
    @safe pure nothrow @nogc
    Vector opBinary(string s)(in float4 scalar) const if (s == "/") {
        return Vector(cast(float4) this.xyzw / cast(float4) scalar);
    }

    /**
     * Multiply a vector by a scalar in-place, changing its value.
     *
     * Params:
     * scalar = A scalar multiplier.
     */
    @safe pure nothrow @nogc
    void opOpAssign(string s)(in float scalar) if (s == "*")
    out {
        assert(this.xyzwArray[3] == 1f);
    } body {
        version(SIMD_X86_OR_X64) {
            this.xyzw *= loadScalar(scalar);
            this.resetW();
        } else {
            this.x *= scalar;
            this.y *= scalar;
            this.z *= scalar;
        }
    }

    /**
     * Divide a vector by a scalar in-place, changing its value.
     *
     * Params:
     * scalar = A scalar divisor.
     */
    @safe pure nothrow @nogc
    void opOpAssign(string s)(in float scalar) if (s == "/")
    out {
        assert(this.xyzwArray[3] == 1f);
    } body {
        version(SIMD_X86_OR_X64) {
            this.xyzw /= loadScalar(scalar);
            this.resetW();
        } else {
            this.x /= scalar;
            this.y /= scalar;
            this.z /= scalar;
        }
    }
}

// Test constructing points and vectors
unittest {
    immutable p = Point(1, 2, 3);
    immutable v = Vector(1, 2, 3);

    assert(p.x == 1f);
    assert(p.y == 2f);
    assert(p.z == 3f);
    assert(v.x == 1f);
    assert(v.y == 2f);
    assert(v.z == 3f);

    Point p2;

    assert(p2.x == 0f);
    assert(p2.y == 0f);
    assert(p2.z == 0f);

    p2.x = -1.1f;
    p2.y = -100.1f;
    p2.z = -1000.1f;

    assert(p2.x == -1.1f);
    assert(p2.y == -100.1f);
    assert(p2.z == -1000.1f);
}

// Test negating a vector.
unittest {
    immutable p = Vector(1, 2, 3);
    immutable q = -p;

    assert(-p.x == q.x);
    assert(-p.y == q.y);
    assert(-p.z == q.z);
}

// Test equality for points.
unittest {
    immutable p = Point(1, 2, 3);
    immutable q = Point(1, 2, 3);

    assert(p == q);

    immutable r = Point(2, 3, 4);

    assert (p != r);
}

// Test equality for vectors.
unittest {
    immutable p = Vector(1, 2, 3);
    immutable q = Vector(1, 2, 3);

    assert(p == q);

    immutable r = Vector(2, 3, 4);

    assert (p != r);
}

// Test adding a vector to a point.
unittest {
    immutable p = Point(1, 2, 3);
    immutable q = Vector(1, 2, 3);

    immutable Point r = p + q;

    assert(r.x == 2f);
    assert(r.y == 4f);
    assert(r.z == 6f);
}

// Test adding a vector to a vector.
unittest {
    immutable p = Vector(1, 2, 3);
    immutable q = Vector(1, 2, 3);

    immutable Vector r = p + q;

    assert(r.x == 2);
    assert(r.y == 4);
    assert(r.z == 6);
}

// Test unary minus for a vector.
unittest {
    immutable v1 = Vector(1, 2, 3);
    immutable v2 = -v1;

    assert(v2.x == -1);
    assert(v2.y == -2);
    assert(v2.z == -3);
}

// Test subtracting two points, yielding a vector.
unittest {
    immutable p = Point(1, 2, 3);
    immutable q = Point(1, 2, 3);

    immutable Vector r = p - q;

    assert(r.x == 0);
    assert(r.y == 0);
    assert(r.z == 0);
}

// Test subtracting two vectors, yielding a vector.
unittest {
    immutable p = Vector(1, 2, 3);
    immutable q = Vector(1, 2, 3);

    immutable Vector r = p - q;

    assert(r.x == 0);
    assert(r.y == 0);
    assert(r.z == 0);
}

// Test scalar multiplication of vectors.
unittest {
    immutable p = Vector(1, 2, 3);

    immutable Vector q = p * 2f;
    immutable Vector r = 2f * p;

    immutable Vector q2 = p * Scalar!2f;
    immutable Vector r2 = Scalar!2f * p;

    assert(q.x == 2);
    assert(q.y == 4);
    assert(q.z == 6);
    assert(q.x == r.x);
    assert(q.y == r.y);
    assert(q.z == r.z);

    assert(q == q2);
    assert(r == r2);
}

// Test the dot product.
unittest {
    immutable p = Vector(1, 2, 3);
    immutable q = Vector(4, 5, 6);

    assert(p * q == 32);
}

// Test vector division.
unittest {
    immutable p = Vector(2, 4, 6);

    immutable Vector q = p / 2f;
    immutable Vector q2 = p / Scalar!2f;

    assert(q.x == 1);
    assert(q.y == 2);
    assert(q.z == 3);

    assert(q == q2);
}

// Test *= with a scalar.
unittest {
    auto p = Vector(1, 2, 3);

    p *= 2;

    assert(p.x == 2);
    assert(p.y == 4);
    assert(p.z == 6);
}

// Test /= with a scalar.
unittest {
    auto p = Vector(2, 4, 6);

    p /= 2;

    assert(p.x == 1);
    assert(p.y == 2);
    assert(p.z == 3);
}

/**
 * Test if two points are approximately equal.
 */
@safe pure nothrow @nogc
bool approxEqual(in Point left, in Point right,
float maxRelDiff=1e-2, float maxAbsDiff=1e-5) {
    return approxEqual(left.x, right.x, maxRelDiff, maxAbsDiff)
        && approxEqual(left.y, right.y, maxRelDiff, maxAbsDiff)
        && approxEqual(left.z, right.z, maxRelDiff, maxAbsDiff);
}

/**
 * Test if two vectors are approximately equal.
 */
@safe pure nothrow @nogc
bool approxEqual(in Vector left, in Vector right,
float maxRelDiff=1e-2, float maxAbsDiff=1e-5) {
    return approxEqual(left.x, right.x, maxRelDiff, maxAbsDiff)
        && approxEqual(left.y, right.y, maxRelDiff, maxAbsDiff)
        && approxEqual(left.z, right.z, maxRelDiff, maxAbsDiff);
}

// Write an alias here so our version of approxEqual overloads correctly.
alias approxEqual = std.math.approxEqual;

// Test approximate equality for points.
unittest {
    immutable p = Point(1, 2, 3);
    immutable q = Point(1, 2, 3);

    assert(approxEqual(p, q));

    immutable r = Point(2, 3, 4);

    assert (!approxEqual(p, r));
}

// Test approximate equality for vectors.
unittest {
    immutable p = Vector(1, 2, 3);
    immutable q = Vector(1, 2, 3);

    assert(approxEqual(p, q));

    immutable r = Vector(2, 3, 4);

    assert (!approxEqual(p, r));
}


// Some of the following unit tests will pretty much repeat the
// implementations, but they are useful for testing the SIMD variations.

/**
 * Compute the square of the length. (a.k.a. norm, magnitude)
 *
 * Compute the norm without applying the square root operation.
 *
 * Returns: The square of the length.
 */
@safe pure nothrow @nogc
float squareOfLength(in Vector vector) {
    return vector * vector;
}

unittest {
    immutable v = Vector(2, 4, 6);

    assert(squareOfLength(v) == 2f ^^ 2 + 4f ^^ 2 + 6f ^^ 2);
}

/**
 * Compute the Euclidean length (a.k.a. norm, magnitude) of this vector.
 *
 * Returns: The length.
 */
@safe pure nothrow @nogc
float length(in Vector vector)  {
    return std.math.sqrt(squareOfLength(vector));
}

unittest {
    immutable v = Vector(2, 4, 6);

    assert(length(v) == std.math.sqrt(2f ^^ 2 + 4f ^^ 2 + 6f ^^ 2));
}

/**
 * Normalize a vector to get a unit vector.
 *
 * Returns: The normalized vector.
 */
@safe pure nothrow @nogc
Vector normalize(in Vector vector) {
    version(SIMD_X86_OR_X64) {
        enum ubyte shiftYZX = 0b00_00_10_01;
        enum ubyte shiftZXY = 0b00_01_00_10;

        // For SIMD, we can keep everything in SIMD types all the way.
        float4 multResult = vector.xyzw * vector.xyzw;

        // Shift the elements around and add three vectors together so we
        // can compute (x,y,z) + (y,z,x) + (z,x,y) and get the elements added
        // together at the right co-ordinates.
        float4 squareResult = multResult
            + shuffle!shiftYZX(multResult)
            + shuffle!shiftZXY(multResult);

        // Apply the square root to every element at once, then
        // use the result to divide every element at once.
        return Vector(
            cast(float4) vector.xyzw
            / cast(float4) __simd(XMM.SQRTPS, squareResult)
        );
    } else {
        return vector / length(vector);
    }
}

unittest {
    immutable v = Vector(2, 4, 6);

    immutable float length = std.math.sqrt(2f ^^ 2 + 4f ^^ 2 + 6f ^^ 2);

    immutable normalizedV = normalize(v);

    assert(normalizedV.x == 2f / length);
    assert(normalizedV.y == 4f / length);
    assert(normalizedV.z == 6f / length);
}

/**
 * Compute the cross product of two vectors.
 *
 * Params:
 * left = The vector on the left.
 * right = The vector on the right.
 *
 * Returns: The product as a new vector.
 */
@safe pure nothrow @nogc
Vector cross(in Vector left, in Vector right) {
    version(SIMD_X86_OR_X64) {
        enum ubyte shiftYZX = 0b00_00_10_01;
        enum ubyte shiftZXY = 0b00_01_00_10;

        return Vector(
            shuffle!shiftYZX(left.xyzw) * shuffle!shiftZXY(right.xyzw)
            - shuffle!shiftZXY(left.xyzw) * shuffle!shiftYZX(right.xyzw)
        );
    } else {
        return Vector(
            left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x
        );
    }
}

unittest {
    immutable v1 = Vector(1, 2, 3);
    immutable v2 = Vector(2, 3, 4);

    immutable expectedCross = Vector(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    );

    immutable actualCross = cross(v1, v2);

    assert(actualCross == expectedCross);
}

/**
 * Compute the distance between two points.
 *
 * Params:
 * left = The point on the left.
 * right = The point on the right.
 *
 * Returns: The distance as a scalar.
 */
@safe pure nothrow @nogc
float distance(in Point left, in Point right) {
    return length(right - left);
}

unittest {
    immutable p = Point(1, 2, 3);
    immutable q = Point(2, 4, 6);

    immutable expectedDistance = std.math.sqrt(
       (1f - 2f) ^^ 2
       + (2f - 4f) ^^ 2
       + (3f - 6f) ^^ 2
    );

    assert(approxEqual(distance(p, q), expectedDistance));
}
