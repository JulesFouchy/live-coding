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

uniform float radius;
uniform vec3 sunDir;
uniform vec3 sunColor;
uniform vec3 bgColor;

uniform float noiseOffset;
uniform float noiseScale;
uniform int octavesNb;

uniform vec3 underwaterColor;
uniform float absorption_coeff;
uniform vec3 absorption_color;
uniform float waterIDR;

uniform float specularStrength;

uniform float landDensity;

uniform float foamFrequency;
uniform float foamDensity;
uniform float foamScale;

uniform float wave_amplitude;
uniform float wave_scale;
uniform float wave_speed;
uniform int wave_octavesNb;

// END DYNAMIC PARAMS

// ----- UsefulConstants ----- //
#define PI 3.14159265358979323846264338327

// ----- Ray marching options ----- //
#define MAX_STEPS 150

#define MAX_DIST 200.
#define SURF_DIST 0.0001
#define NORMAL_DELTA 0.0001

#define FBM_MAX_ITER 10

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


mat3 rot3X(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat3(
        1., 0., 0.,
        0., c,  -s,
        0., s,  c
    );
}

mat3 rot3Y(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat3(
        c, 0., -s,
        0., 1., 0,
        s, 0., c
    );
}
mat3 rot3Z(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat3(
        c, -s, 0.,
        s, c, 0.,
        0., 0., 1.
    );
}

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

float perlinNoise(vec2 x) {
    vec2 id = floor(x);
    vec2 f = fract(x);

	float a = hash1(id);
    float b = hash1(id + vec2(1.0, 0.0));
    float c = hash1(id + vec2(0.0, 1.0));
    float d = hash1(id + vec2(1.0, 1.0));
	// Same code, with the clamps in smoothstep and common subexpressions
	// optimized away.
    vec2 u = hermiteInter(f);
	return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
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


vec4 mod289(vec4 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x)
{
  return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec4 fade(vec4 t) {
  return t*t*t*(t*(t*6.0-15.0)+10.0);
}

// Classic Perlin noise
float perlinNoise(vec4 P)
{
  vec4 Pi0 = floor(P); // Integer part for indexing
  vec4 Pi1 = Pi0 + 1.0; // Integer part + 1
  Pi0 = mod289(Pi0);
  Pi1 = mod289(Pi1);
  vec4 Pf0 = fract(P); // Fractional part for interpolation
  vec4 Pf1 = Pf0 - 1.0; // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = vec4(Pi0.zzzz);
  vec4 iz1 = vec4(Pi1.zzzz);
  vec4 iw0 = vec4(Pi0.wwww);
  vec4 iw1 = vec4(Pi1.wwww);

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);
  vec4 ixy00 = permute(ixy0 + iw0);
  vec4 ixy01 = permute(ixy0 + iw1);
  vec4 ixy10 = permute(ixy1 + iw0);
  vec4 ixy11 = permute(ixy1 + iw1);

  vec4 gx00 = ixy00 * (1.0 / 7.0);
  vec4 gy00 = floor(gx00) * (1.0 / 7.0);
  vec4 gz00 = floor(gy00) * (1.0 / 6.0);
  gx00 = fract(gx00) - 0.5;
  gy00 = fract(gy00) - 0.5;
  gz00 = fract(gz00) - 0.5;
  vec4 gw00 = vec4(0.75) - abs(gx00) - abs(gy00) - abs(gz00);
  vec4 sw00 = step(gw00, vec4(0.0));
  gx00 -= sw00 * (step(0.0, gx00) - 0.5);
  gy00 -= sw00 * (step(0.0, gy00) - 0.5);

  vec4 gx01 = ixy01 * (1.0 / 7.0);
  vec4 gy01 = floor(gx01) * (1.0 / 7.0);
  vec4 gz01 = floor(gy01) * (1.0 / 6.0);
  gx01 = fract(gx01) - 0.5;
  gy01 = fract(gy01) - 0.5;
  gz01 = fract(gz01) - 0.5;
  vec4 gw01 = vec4(0.75) - abs(gx01) - abs(gy01) - abs(gz01);
  vec4 sw01 = step(gw01, vec4(0.0));
  gx01 -= sw01 * (step(0.0, gx01) - 0.5);
  gy01 -= sw01 * (step(0.0, gy01) - 0.5);

  vec4 gx10 = ixy10 * (1.0 / 7.0);
  vec4 gy10 = floor(gx10) * (1.0 / 7.0);
  vec4 gz10 = floor(gy10) * (1.0 / 6.0);
  gx10 = fract(gx10) - 0.5;
  gy10 = fract(gy10) - 0.5;
  gz10 = fract(gz10) - 0.5;
  vec4 gw10 = vec4(0.75) - abs(gx10) - abs(gy10) - abs(gz10);
  vec4 sw10 = step(gw10, vec4(0.0));
  gx10 -= sw10 * (step(0.0, gx10) - 0.5);
  gy10 -= sw10 * (step(0.0, gy10) - 0.5);

  vec4 gx11 = ixy11 * (1.0 / 7.0);
  vec4 gy11 = floor(gx11) * (1.0 / 7.0);
  vec4 gz11 = floor(gy11) * (1.0 / 6.0);
  gx11 = fract(gx11) - 0.5;
  gy11 = fract(gy11) - 0.5;
  gz11 = fract(gz11) - 0.5;
  vec4 gw11 = vec4(0.75) - abs(gx11) - abs(gy11) - abs(gz11);
  vec4 sw11 = step(gw11, vec4(0.0));
  gx11 -= sw11 * (step(0.0, gx11) - 0.5);
  gy11 -= sw11 * (step(0.0, gy11) - 0.5);

  vec4 g0000 = vec4(gx00.x,gy00.x,gz00.x,gw00.x);
  vec4 g1000 = vec4(gx00.y,gy00.y,gz00.y,gw00.y);
  vec4 g0100 = vec4(gx00.z,gy00.z,gz00.z,gw00.z);
  vec4 g1100 = vec4(gx00.w,gy00.w,gz00.w,gw00.w);
  vec4 g0010 = vec4(gx10.x,gy10.x,gz10.x,gw10.x);
  vec4 g1010 = vec4(gx10.y,gy10.y,gz10.y,gw10.y);
  vec4 g0110 = vec4(gx10.z,gy10.z,gz10.z,gw10.z);
  vec4 g1110 = vec4(gx10.w,gy10.w,gz10.w,gw10.w);
  vec4 g0001 = vec4(gx01.x,gy01.x,gz01.x,gw01.x);
  vec4 g1001 = vec4(gx01.y,gy01.y,gz01.y,gw01.y);
  vec4 g0101 = vec4(gx01.z,gy01.z,gz01.z,gw01.z);
  vec4 g1101 = vec4(gx01.w,gy01.w,gz01.w,gw01.w);
  vec4 g0011 = vec4(gx11.x,gy11.x,gz11.x,gw11.x);
  vec4 g1011 = vec4(gx11.y,gy11.y,gz11.y,gw11.y);
  vec4 g0111 = vec4(gx11.z,gy11.z,gz11.z,gw11.z);
  vec4 g1111 = vec4(gx11.w,gy11.w,gz11.w,gw11.w);

  vec4 norm00 = taylorInvSqrt(vec4(dot(g0000, g0000), dot(g0100, g0100), dot(g1000, g1000), dot(g1100, g1100)));
  g0000 *= norm00.x;
  g0100 *= norm00.y;
  g1000 *= norm00.z;
  g1100 *= norm00.w;

  vec4 norm01 = taylorInvSqrt(vec4(dot(g0001, g0001), dot(g0101, g0101), dot(g1001, g1001), dot(g1101, g1101)));
  g0001 *= norm01.x;
  g0101 *= norm01.y;
  g1001 *= norm01.z;
  g1101 *= norm01.w;

  vec4 norm10 = taylorInvSqrt(vec4(dot(g0010, g0010), dot(g0110, g0110), dot(g1010, g1010), dot(g1110, g1110)));
  g0010 *= norm10.x;
  g0110 *= norm10.y;
  g1010 *= norm10.z;
  g1110 *= norm10.w;

  vec4 norm11 = taylorInvSqrt(vec4(dot(g0011, g0011), dot(g0111, g0111), dot(g1011, g1011), dot(g1111, g1111)));
  g0011 *= norm11.x;
  g0111 *= norm11.y;
  g1011 *= norm11.z;
  g1111 *= norm11.w;

  float n0000 = dot(g0000, Pf0);
  float n1000 = dot(g1000, vec4(Pf1.x, Pf0.yzw));
  float n0100 = dot(g0100, vec4(Pf0.x, Pf1.y, Pf0.zw));
  float n1100 = dot(g1100, vec4(Pf1.xy, Pf0.zw));
  float n0010 = dot(g0010, vec4(Pf0.xy, Pf1.z, Pf0.w));
  float n1010 = dot(g1010, vec4(Pf1.x, Pf0.y, Pf1.z, Pf0.w));
  float n0110 = dot(g0110, vec4(Pf0.x, Pf1.yz, Pf0.w));
  float n1110 = dot(g1110, vec4(Pf1.xyz, Pf0.w));
  float n0001 = dot(g0001, vec4(Pf0.xyz, Pf1.w));
  float n1001 = dot(g1001, vec4(Pf1.x, Pf0.yz, Pf1.w));
  float n0101 = dot(g0101, vec4(Pf0.x, Pf1.y, Pf0.z, Pf1.w));
  float n1101 = dot(g1101, vec4(Pf1.xy, Pf0.z, Pf1.w));
  float n0011 = dot(g0011, vec4(Pf0.xy, Pf1.zw));
  float n1011 = dot(g1011, vec4(Pf1.x, Pf0.y, Pf1.zw));
  float n0111 = dot(g0111, vec4(Pf0.x, Pf1.yzw));
  float n1111 = dot(g1111, Pf1);

  vec4 fade_xyzw = fade(Pf0);
  vec4 n_0w = mix(vec4(n0000, n1000, n0100, n1100), vec4(n0001, n1001, n0101, n1101), fade_xyzw.w);
  vec4 n_1w = mix(vec4(n0010, n1010, n0110, n1110), vec4(n0011, n1011, n0111, n1111), fade_xyzw.w);
  vec4 n_zw = mix(n_0w, n_1w, fade_xyzw.z);
  vec2 n_yzw = mix(n_zw.xy, n_zw.zw, fade_xyzw.y);
  float n_xyzw = mix(n_yzw.x, n_yzw.y, fade_xyzw.x);
  return 2.2 * n_xyzw;
}

float fbm (vec2 x, float H, int octaves) {
	float G = exp2(-H);
	float v = 0.;
	float f = 1.;
	float amp = 1.;
	float aSum = 1.;
	
    vec2 shift = vec2(100.);
	for ( int i=0; i < FBM_MAX_ITER; ++i) {
		if( i >= octaves) break;
		v += amp * perlinNoise(f*x);
		f *= 2.;
		amp *= G;
		aSum += amp;
		// Rotate and shift to reduce axial bias
		x = rot2(0.5) * x + shift;
	}
	return v / aSum;
}

float fbm (vec3 x, float H, int octaves) {
	float G = exp2(-H);
	float v = 0.;
	float f = 1.;
	float amp = 1.;
	float aSum = 1.;
	
	for ( int i=0; i < FBM_MAX_ITER; ++i) {
		if( i >= octaves) break;
		v += amp * perlinNoise(f*x);
		f *= 2.;
		amp *= G;
		aSum += amp;
	}
	return v / aSum;
}

float fbm (vec4 x, float H, int octaves) {
	float G = exp2(-H);
	float v = 0.;
	float f = 1.;
	float amp = 1.;
	float aSum = 1.;
	
	for ( int i=0; i < FBM_MAX_ITER; ++i) {
		if( i >= octaves) break;
		v += amp * perlinNoise(f*x);
		f *= 2.;
		amp *= G;
		aSum += amp;
	}
	return v / aSum;
}

vec3 fbm (float x, float H, int octaves) {
	float G = exp2(-H);
	vec3 v = vec3(0.);
	float f = 1.;
	float amp = 1.;
	float aSum = 1.;
	
	for ( int i=0; i < FBM_MAX_ITER; ++i) {
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

float sphereSDF(vec3 pos) {

    vec3 sphereNormal = normalize(pos);

    vec2 sphereUV = asin(sphereNormal.xy)/PI + 0.5;
    
    
    return length(pos) - radius + wave_amplitude * mix(-1., landDensity, fbm(vec4(pos * wave_scale, uTime * wave_speed), 0.7, wave_octavesNb));
}

float noisySphereSDF(vec3 pos) {
    return length(pos) - radius - mix(-1., landDensity, fbm(pos * noiseScale + noiseOffset, 1., octavesNb));
}

float sceneSDF(vec3 pos) {
    return min(sphereSDF(pos), noisySphereSDF(pos));
}

/*
vec3 getNormal(vec3 p) {
  const float h = NORMAL_DELTA;
	const vec2 k = vec2(1., -1.);
    return normalize( k.xyy * sceneSDF( p + k.xyy*h ) + 
                      k.yyx * sceneSDF( p + k.yyx*h ) + 
                      k.yxy * sceneSDF( p + k.yxy*h ) + 
                      k.xxx * sceneSDF( p + k.xxx*h ) );
}*/

#define DECL_NORMAL_FUNC(_name, _sceneSDF, _h) vec3 _name(vec3 p) {const vec2 k = vec2(1., -1.); return normalize( k.xyy * _sceneSDF( p + k.xyy*_h ) + k.yyx * _sceneSDF( p + k.yyx*_h ) + k.yxy * _sceneSDF( p + k.yxy*_h ) + k.xxx * _sceneSDF( p + k.xxx*_h ) ); }

DECL_NORMAL_FUNC(getNormalSurface, sceneSDF, NORMAL_DELTA)
//DECL_NORMAL_FUNC(getNormalWithDepth, sceneSDF, NORMAL_DELTA)

#define DECL_RAYMARCH_FUNC(_name, _sceneSDF, _precision) float _name(vec3 ro, vec3 rd) { float t = 0.; for(int i = 0; i < MAX_STEPS; i++) { vec3 pos = ro + rd * t; float d = _sceneSDF(pos); t += _precision * d; if( t > MAX_DIST || abs(d) < SURF_DIST*0.99) break; } return t; }

DECL_RAYMARCH_FUNC(rayMarchSurface, sceneSDF, 0.8)
DECL_RAYMARCH_FUNC(rayMarchWithDepth, noisySphereSDF, 0.8)

float distanceInsideSphere(vec3 r0, vec3 rd) {
    // - r0: ray origin
    // - rd: normalized ray direction
    // - s0: sphere center
    // - sr: sphere radius
    // - Returns distance between the two intersections
    float a = dot(rd, rd);
    float b = 2.0 * dot(rd, r0);
    float c = dot(r0, r0) - (radius * radius);
    float delta = b*b - 4.0*a*c;
    float t = (-b + sqrt(abs(delta))) / 2. / a;
    return sqrt(abs(delta)) / a;
}


vec3 render(vec3 ro, vec3 rd) { // ray origin and dir
  

  vec3 waterColor = vec3(.015, .110, .455);
  vec3 grassColor = vec3(.086, .132, .018);
  vec3 beachColor = vec3(.353, .372, .121);
  vec3 rockColor = vec3(.080, .050, .030);
  vec3 snowColor = vec3(.6, .6, .6);
  vec3 skyColor = vec3(117, 164, 250) / 255;

  vec3 finalCol = skyColor;

  float d = rayMarchSurface(ro, rd);
  vec2 sphereUV;
  if( d < MAX_DIST) {
    vec3 p = ro + rd * d;
    float dist = length(p) - radius;

    vec3 sphereNormal = normalize(p);
    sphereUV = asin(sphereNormal.xy)/PI + 0.5;
    sphereUV = fract(sphereUV*40);
    // water
    // if (dist < 0.001) {
    if(sphereSDF(p) < noisySphereSDF(p)) {

        vec3 normal = getNormalSurface(p);

        vec3 underwater_rd = rd;

        float naturalDepth = rayMarchWithDepth(p, normalize(-p));

        underwater_rd = refract(underwater_rd, normal, waterIDR);
        float underwaterDistance = rayMarchWithDepth(p, underwater_rd);

        vec3 colorBehindWater;
        if(underwaterDistance > MAX_DIST) {
            underwaterDistance = distanceInsideSphere(p, underwater_rd);
            colorBehindWater = skyColor;
        }else {
            colorBehindWater = underwaterColor;
        }

        vec3 underwaterPt = p + underwater_rd * underwaterDistance;
        // vec3 absorptionFromSun = exp(-absorption_coeff * absorption_color * distanceInsideSphere(underwaterPt, sunDir));
        vec3 absorptiontoEyes = exp(-absorption_coeff * absorption_color * underwaterDistance);
        // finalCol = (sunColor - absorptionFromSun) * underwaterColor * absorptiontoEyes;
        
        finalCol = sunColor * colorBehindWater * absorptiontoEyes;

        // REFLEXION
        vec3 reflectDir = reflect(rd, normal);

        vec3 H = normalize(sunDir + -reflectDir);
        float reflection = pow(saturate(dot(normal, H)), specularStrength);

        //Sum up the specular light factoring
        finalCol += reflection * sunColor;

        // FOAM    
        float foamNoise = fbm(p * 10  * foamScale + mod(uTime, 1000) * foamFrequency * vec3(2, 4, 1.8), 1, 5);
        float foarCoeff = saturate(exp(-naturalDepth*10/foamDensity));
        
        finalCol = mix(finalCol, vec3(0.8), foamNoise*foarCoeff);
        
    }
    else {
    
        vec3 normal = getNormalSurface(p);
        // vec3 ref = reflect(rd, normal);


        float sunLight = saturate(dot(normal, normalize(sunDir)));
        
        // float sunSpecular = pow(saturate(dot(normalize(sunDir), ref)), specularStrength); // Phong
        
        finalCol = mix(waterColor, beachColor, smoothstep(0., 0.01, dist));
        finalCol = mix(finalCol, grassColor, smoothstep(0.04, 0.1, dist));
        finalCol = mix(finalCol, rockColor, smoothstep(0.2, 0.4, dist));
        if(dot(normal, normalize(p)) > 0.85){
            finalCol = mix(finalCol, snowColor, smoothstep(0.42, 0.5, dist));
        }
        // if(dist < 0.) {
        //     finalCol = mix(finalCol, waterColor, pow(-dist, 1./5.));
        // }
        float illumination = 0.2 + sunLight * 0.8;
        illumination += sunLight; //vec3(0., 1., 0.);
        finalCol *= saturate(illumination);
        finalCol *= sunColor;
        //finalCol = vec3(perlinNoise(noiseOffset + p*noiseScale));
    }
  }
  // finalCol = vec3(sphereUV, 0);

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
    col = pow(max(col, vec3(0.)), vec3(.4545)); // gamma correction
    gl_FragColor = vec4(col, 1.0);
}