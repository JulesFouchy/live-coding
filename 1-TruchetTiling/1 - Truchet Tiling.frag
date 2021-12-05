#version 430

const float TAU = 6.283185307179586476925286766559;

in vec2 vTexCoords;
uniform float uAspectRatio;
uniform vec3 uCamX;
uniform vec3 uCamY;
uniform vec3 uCamZ;
uniform vec3 uCamPos;
uniform float uFocalLength;

uniform float uTime;

// BEGIN DYNAMIC PARAMS
uniform vec3 _color;
uniform int nb;
uniform float noise_scale;
uniform vec2 offset;
uniform float radius; 
// END DYNAMIC PARAMS

float disk(vec2 uv, vec2 center, float radius) {
    return length(uv - center) < radius ? 1. : 0.;
}

float hash1(vec2 p) {
    p = fract(p * vec2(155.65, 121.56));
    p += dot(p, p + 32.45);
    return fract(p.x * p.y);
}

vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

float snoise(vec2 v){
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
           -0.577350269189626, 0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
  + i.x + vec3(0.0, i1.x, 1.0 ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
    dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

float perlinNoise(vec2 p) {
    vec2 guv = fract(p);
    vec2 gid = floor(p);
    float a = hash1(gid);
    float b = hash1(gid + vec2(1., 0.));
    float c = hash1(gid + vec2(0., 1.));
    float d = hash1(gid + vec2(1., 1.));
    
    guv = smoothstep(0, 1, guv);
    return mix(mix(a, b, guv.x), mix(c, d, guv.x), guv.y);
}


void main() {
    vec2 uv = vTexCoords;
    uv.x *= uAspectRatio;
    // uv += uTime/20;

    vec2 guv = fract(uv*nb);
    const vec2 gid = floor(uv*nb);
    
    vec3 col = vec3( 0.);
    // float d = length(gid - nb/2);
    // d *= d;

    // col = vec3(1.) * disk(guv, vec2(0.5), mix(
    //     0.,
    //     0.025 * (nb-d),
    //     cos(uTime + hash1(gid))/2+0.5//hash1(gid)
    // ));
    // col = vec3(1.) * disk(guv, vec2(0.5), (snoise(gid*noise_scale)*0.5+0.5)*0.5 );
    // col = vec3(snoise(uv*10)*0.5+0.5);
    // // float d = dot(guv, vec2(1, -1));

    guv -= 0.5;
    
    // if(guv.x > 0.48 || guv.y > 0.48) col = vec3(1, 0, 0);
    
    // //flip
    if( hash1(gid*noise_scale) < 0.5) {
        guv.x *= -1.;
    }

    // symmetry
    vec2 sguv = guv;
    if (sguv.x > sguv.y) {
        sguv = sguv.yx;
    }

    // Line
    float d;
    float r = hash1((gid + vec2(11, 3))*noise_scale);
    // if (r < 0.5) {
    //     d = abs(sguv.x - sguv.y + 0.5);
    // }
    if (r < 0.5) {
        d = length(sguv + vec2(0.5, -0.5));
        d = abs(d-0.5);
    }
    else {
        d = min(abs(guv.x), abs(guv.y));
    }
    d = 1. - smoothstep(0., 0.001, d-radius);

    col += vec3(d);
    
    gl_FragColor = vec4(col, 1.);
}