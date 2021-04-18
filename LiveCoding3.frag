#version 430

in vec2 vTexCoords;
uniform float uAspectRatio;
uniform vec3 uCamX;
uniform vec3 uCamY;
uniform vec3 uCamZ;
uniform vec3 uCamPos;
uniform float uFocalLength;

uniform float uTime;

// BEGIN DYNAMIC PARAMS

uniform vec3 sunDir;
uniform vec3 sunColor;
uniform vec3 ambientColor;
uniform vec3 backgroundColor1;
uniform vec3 backgroundColor2;
uniform float specularStrength;
uniform float smoothing;
uniform float torusHeight;
uniform float glow0FallOff;
uniform vec3 glowColor;

// END DYNAMIC PARAMS

// ----- UsefulConstants ----- //
#define PI 3.14159265358979323846264338327

// ----- Ray marching options ----- //
#define MAX_STEPS 150

#define MAX_DIST 200.
#define SURF_DIST 0.0001
#define NORMAL_DELTA 0.0001

// ----- Useful functions ----- //
#define rot2(a) mat2(cos(a), -sin(a), sin(a), cos(a))
float maxComp(vec2 v) { return max(v.x , v.y); }
float maxComp(vec3 v) { return max(max(v.x , v.y), v.z); }
float cro(vec2 a,vec2 b) { return a.x*b.y - a.y*b.x; }
float mult(vec2 v) { return v.x*v.y; }
float mult(vec3 v) { return v.x*v.y*v.z; }
float sum(vec2 v) { return v.x+v.y; }
float sum(vec3 v) { return v.x+v.y+v.z; }
#define map(t, a, b) a + t * (b - a)
#define saturate(v) clamp(v, 0., 1.)
#define hermiteInter(t) t * t * (3.0 - 2.0 * t)

// ----- Noise functions ----- //
float hash1(float p) {
    p = fract(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float hash1(vec2 p) {
	vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float hash1(vec3 p3) {
	p3  = fract(p3 * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 hash3(float p) {
   vec3 p3 = fract(vec3(p) * vec3(.1031, .1030, .0973));
   p3 += dot(p3, p3.yzx+33.33);
   return fract((p3.xxy+p3.yzz)*p3.zyx); 
}

float perlinNoise(float x) {
    float id = floor(x);
    float f = fract(x);
    float u = hermiteInter(f);
    return mix(hash1(id), hash1(id + 1.0), u);
}

vec3 perlinNoise3(float x) {
    float id = floor(x);
    float f = fract(x);
    float u = hermiteInter(f);
    return mix(hash3(id), hash3(id + 1.0), u);
}

float perlinNoise(vec3 x) {
    const vec3 step = vec3(110., 241., 171.);

    vec3 id = floor(x);
    vec3 f = fract(x);
 
    // For performance, compute the base input to a 1D hash from the integer part of the argument and the 
    // incremental change to the 1D based on the 3D -> 1D wrapping
    float n = dot(id, step);

    vec3 u = hermiteInter(f);
    return mix(mix(mix( hash1(n + dot(step, vec3(0, 0, 0))), hash1(n + dot(step, vec3(1, 0, 0))), u.x),
                   mix( hash1(n + dot(step, vec3(0, 1, 0))), hash1(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
               mix(mix( hash1(n + dot(step, vec3(0, 0, 1))), hash1(n + dot(step, vec3(1, 0, 1))), u.x),
                   mix( hash1(n + dot(step, vec3(0, 1, 1))), hash1(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}

vec3 fbm (float x, float H, int octaves) {
	float G = exp2(-H);
	vec3 v = vec3(0.);
	float f = 1.;
	float amp = 1.;
	float aSum = 1.;
	
	for ( int i=0; i < 10; ++i) {
		if( i >= octaves) break;
		v += amp * perlinNoise3(f*x);
		f *= 2.;
		amp *= G;
		aSum += amp;
	}
	return v / aSum;
}

vec3 vectorWiggle(float x) {
    return fbm(x, 1., 2);
}

vec3 blendColor(float t, vec3 a, vec3 b) {
	return sqrt((1. - t) * pow(a, vec3(2.)) + t * pow(b, vec3(2.)));
}

// ----- distance functions for 3D primitives ----- //
// source: http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

float smin( float a, float b, float k ) {
    float h = max(k-abs(a-b), 0.0)/k;
    return min(a, b) - h*h*0.25;
}

float polysmin( float a, float b, float k ) {
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float cylSDF(vec2 p, float radius) { return length(p) - radius; }

float boxSDF(vec3 p, vec3 boxDim) {
  vec3 q = abs(p) - boxDim;
  return length(max(q, 0.0)) + min(maxComp(q), 0.0);
}

float sdTorus(vec3 p, vec2 t) {
  vec2 q = vec2(length(p.xz)-t.x, p.y);
  return length(q) - t.y;
}

float sphereSDF(vec3 p, float radius) { return length(p) - radius;}

float sceneSDF(vec3 pos) {

  float d = MAX_DIST;

  float spheredist = sphereSDF(pos, 5.);
  float torusDis = sdTorus(pos - vec3(0., torusHeight, 0.), vec2(5., 1.));
  
  d = polysmin(d, spheredist, smoothing);
  d = max(-d, torusDis);

  for ( float i=0; i < 10; ++i) {
    d = min(d, sdTorus(pos - (vectorWiggle(i + uTime/2)*2-1)*10, vec2(perlinNoise(map(i*1.12, 1, 1)), 0.4)));
  }
  return d;
}

vec3 getNormal(vec3 p) {
  const float h = NORMAL_DELTA;
	const vec2 k = vec2(1., -1.);
    return normalize( k.xyy * sceneSDF( p + k.xyy*h ) + 
                      k.yyx * sceneSDF( p + k.yyx*h ) + 
                      k.yxy * sceneSDF( p + k.yxy*h ) + 
                      k.xxx * sceneSDF( p + k.xxx*h ) );
}

float glow = 0;
float marchingCount = 0;

float rayMarching(vec3 ro, vec3 rd) {
    float t = 0.;
 	
    for(int i = 0; i < MAX_STEPS; i++) {
    	vec3 pos = ro + rd * t;
      float d = sceneSDF(pos);
      
      t += d;

      // If we are very close to the object, consider it as a hit and exit this loop
      if( t > MAX_DIST || abs(d) < SURF_DIST*0.99) break;

      marchingCount += 1;
      glow += glow0FallOff / (d * d + glow0FallOff); // with gaussian fallOff
    }
    return t;
}

vec3 bgColor(vec3 rd) {

  float k = rd.y*0.5+0.5;
  return mix(backgroundColor1, backgroundColor2, k);
}

vec3 render(vec3 ro, vec3 rd) { // ray origin and dir
  
  vec3 orangeColor = vec3(255,166,43)/255.0;

  // reset glow
  glow = 0.; //MAX_DIST * 100.;
  vec3 nSunDir = normalize(sunDir);

  vec3 finalCol = bgColor(rd);

  float d = rayMarching(ro, rd);

  if( d < MAX_DIST) {
    vec3 p = ro + rd * d;
    vec3 normal = getNormal(p); 
    vec3 ref = reflect(rd, normal);
    
    float sunFactor = saturate(dot(normal, nSunDir));

    float sunSpecular = pow(saturate(dot(nSunDir, ref)), specularStrength); // Phong

    finalCol = vec3(sunFactor) + sunSpecular * sunColor + ambientColor * 0.5 ;

    finalCol -= smoothstep(20, 30, marchingCount);
  }
  else {
    finalCol = blendColor(glow*0.1, finalCol, glowColor);
    //finalCol += glowColor * glow * 0.1;
  }

  // finalCol = vec3(glow*0.1);
	return vec3(saturate(finalCol));
}

void main() {
    // Setup camera
    vec3 ro = uCamPos;
    vec3 rd = normalize(
              uCamX * (vTexCoords.x - 0.5) * uAspectRatio
            + uCamY * (vTexCoords.y - 0.5)
            - uCamZ * uFocalLength
    );
    
    vec3 col = render(ro , rd);
    // col = pow(max(col, vec3(0.)), vec3(.4545)); // gamma correction
    gl_FragColor = vec4(col, 1.0);
}