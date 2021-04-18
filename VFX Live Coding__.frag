#version 430

in vec2 vTexCoords;
uniform float uAspectRatio;
uniform vec3 uCamX;
uniform vec3 uCamY;
uniform vec3 uCamZ;
uniform vec3 uCamPos;
uniform float uFocalLength;

uniform float uTime;

struct Ray {
  vec3 origin;
  vec3 direction;
};

struct Plane {
  vec3 origin;
  vec3 normal;
};

void main()
{
    // Setup camera
    Ray ray = {
      uCamPos,
      normalize(
        uCamX * (vTexCoords.x - 0.5) * uAspectRatio
      + uCamY * (vTexCoords.y - 0.5)
      - uCamZ * uFocalLength
    )};
    //
    vec3 col = vec3(0.);
    Plane plane = {
      vec3(0.),
      vec3(0., 1., 0.)
    };
    const vec3 xAxis = vec3(-plane.normal.y, plane.normal.x, 0.);
    const vec3 yAxis = cross(plane.normal, xAxis);
    float t = dot(plane.origin - ray.origin, plane.normal) / dot(ray.direction, plane.normal);
    if (t > 0) {
      const vec3 P = ray.origin + t * ray.direction;
      const vec2 uv = 0.01 * vec2(
        dot(xAxis, P),
        dot(yAxis, P)
      );
      if (length(uv) < 1. && length(uv) > 0.8)
        col = vec3(0., 0.1, 0.8);
    }
    gl_FragColor = vec4(col, 1.);
}