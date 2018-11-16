#version 330

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec3 vertexCanonicalPosition;
layout (location = 2) in vec2 vertexTexCoord;
layout (location = 3) in vec3 vertexNormal;

out vec3 fragPosition;
out vec3 fragCanonicalPosition;
out vec3 fragPositionCam;
out vec2 fragTexCoord;
out vec3 fragNormal;

void main()
{
    fragPosition = vertexPosition;
    fragCanonicalPosition = vertexCanonicalPosition;
    fragTexCoord = vertexTexCoord;

    vec3 rawNormal = vec3(modelViewMatrix*vec4(vertexNormal,0.0));
    float rawLength = length(rawNormal);
    if (rawLength > 0) {
        fragNormal = rawNormal / rawLength;
    } else {
        fragNormal = rawNormal;
    }

    vec4 vertexPositionCam = modelViewMatrix*vec4(vertexPosition,1.0);

    fragPositionCam = vec3(vertexPositionCam);

    gl_Position = projectionMatrix*vertexPositionCam;
}
