#include "stdafx.h"
#include <iostream>

#include <Windows.h>                // getTickCount64
#include <unordered_map>

#include "GLFW/glfw3.h"
#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"

#include "RenderModule/vulkan.h"

#define RANDOMNUM 8

using namespace RS;

void mouseMove(GLFWwindow* window, double xs, double ys);
void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods);

//////////////////////////

const uint32_t WIDTH = 1200;
const uint32_t HEIGHT = 800;
GLFWwindow* window = nullptr;

#include <random>
#define PI 3.141592653589793  

uint32_t getNewSeed(uint32_t param1, uint32_t param2, uint32_t numPermutation)
{
    uint32_t s0 = 0;
    uint32_t v0 = param1;
    uint32_t v1 = param2;

    for (uint32_t perm = 0; perm < numPermutation; perm++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

const float rng(uint32_t& seed)
{
    seed = (1664525u * seed + 1013904223u);
    return ((double)(seed & 0x00FFFFFF) / (double)0x01000000);
}

std::uniform_real_distribution<double> dist(0.0, 1.0);
std::mt19937 gen(std::random_device{}());
uint32_t seed = getNewSeed(10, 10, 10);
//#define SAMPLE dist(gen)
#define SAMPLE rng(seed)

float3 sample_upper_sphere_HG()
{
    float3 sample(0, 0, 0);
    for (;;) {
        sample = {
            SAMPLE * 2.0f - 1.0f,
            SAMPLE * 2.0f - 1.0f,
            SAMPLE * 2.0f - 1.0f
        };
        if (sample._x * sample._x + sample._y * sample._y + sample._z * sample._z < 1.0)
            break;
    }

    double norm = std::sqrt(sample._x * sample._x + sample._y * sample._y + sample._z * sample._z);
    sample._x /= norm;
    sample._y /= norm;
    sample._z /= norm;

    if (sample._z < 0.0)
        sample._z *= -1.0;

    return sample;
}

#pragma region Shader
const char* raygen_src = R"(
#version 460
#extension GL_EXT_ray_tracing : enable

layout(binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 1, rgba8) uniform image2D outImage;
layout(binding = 2) uniform CameraProperties 
{
    vec3 cameraPos;
    float yFov_degree;
    float _time;
} g;

struct Payload
{
    uint depth;
    vec3 hitValue;
};

layout(location = 0) rayPayloadEXT Payload payload;

const float tMax = 100.0;

void main() 
{
    const vec3 cameraX = vec3(1, 0, 0);
    const vec3 cameraY = vec3(0, -1, 0);
    const vec3 cameraZ = vec3(0, 0, -1);
    const float aspect_y = tan(radians(g.yFov_degree) * 0.5);
    const float aspect_x = aspect_y * float(gl_LaunchSizeEXT.x) / float(gl_LaunchSizeEXT.y);

    const vec2 screenCoord = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 ndc = screenCoord/vec2(gl_LaunchSizeEXT.xy) * 2.0 - 1.0;
    vec3 rayDir = normalize(ndc.x*aspect_x*cameraX + ndc.y*aspect_y*cameraY + cameraZ);

    payload.depth = 2;
    payload.hitValue = vec3(0.0);

    traceRayEXT(
        topLevelAS,                         // topLevel
        gl_RayFlagsOpaqueEXT, 0xff,         // rayFlags, cullMask
        0, 1, 0,                            // sbtRecordOffset, sbtRecordStride, missIndex
        g.cameraPos, 0.0, rayDir, tMax,     // origin, tmin, direction, tmax
        0);                                 // payload

    imageStore(outImage, ivec2(gl_LaunchIDEXT.xy), vec4(payload.hitValue, 0.0));
})";

const char* miss_src1 = R"(
#version 460
#extension GL_EXT_ray_tracing : enable

struct Payload
{
    uint depth;
    vec3 hitValue;
};

layout(location = 0) rayPayloadInEXT Payload payload;

void main()
{
    payload.hitValue = vec3(1.0);
})";

const char* chit_src1 = R"(
#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#define RANDOMNUM 128

#define TRUE 1
#define FALSE 0

// Information of a obj model when referenced in a shader
struct ObjDesc
{
  uint64_t vertexAddress;         // Address of the Vertex buffer
  uint64_t indexAddress;          // Address of the index buffer
};

layout(shaderRecordEXT) buffer CustomData
{
    vec3 color;
    uint isLight;
    ObjDesc objDesc;
    mat4 trans;
};

struct Vertex  // See ObjLoader, copy of VertexObj, could be compressed for device
{
  vec3 pos;
  float padding1;
  vec3 nrm;
  float padding2;
  vec2 texCoord;
  vec2 padding3;
};

struct RandomRay 
{
    vec3 randomRay[RANDOMNUM];
    float padding;
};

layout(binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(buffer_reference, scalar) buffer Vertices {Vertex v[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer Indices {ivec3 i[]; }; // Triangle indices
//layout(set = 1, binding = 0, scalar) buffer RandomRay_ { RandomRay i[]; } rR;

struct Payload
{
    uint depth;
    vec3 hitValue;
};

layout(location = 0) rayPayloadInEXT Payload payload;
hitAttributeEXT vec2 attribs;

float RandomValue(inout uint state) {
    state *= (state + 195439) * (state + 124395) * (state + 845921);
    return state / 4294967295.0;
}
float RandomValueNormalDistribution(inout uint state) {
    float theta = 2 * 3.1415926 * RandomValue(state);
    float rho = sqrt(-2 * log(RandomValue(state)));
    return rho * cos(theta);
}
vec3 RandomDirection(inout uint state) {
    float x, y, z;
    for( ; ; ){
        x = RandomValue(state) * 2.0 - 1.0;
        y = RandomValue(state) * 2.0 - 1.0;
        z = RandomValue(state) * 2.0 - 1.0;

        if(dot(vec3(x, y, z), vec3(x, y, z)) > 1.0)
            continue;
        else
            break;
    }
    return normalize(vec3(x, y, z));
}
vec3 RandomHemisphereDirection(vec3 normal, inout uint state) {
    vec3 dir = RandomDirection(state);
    return dir * sign(dot(normal, dir));
}

void main()
{
    if(payload.depth == 0){
        if(isLight == TRUE){                // => 발광체
            payload.hitValue = vec3(1.0) * 100;//vec3(1.0);
        }
        else{                               // => 비발광체
            payload.hitValue = vec3(0.0);
        }
        return;
    }

    ObjDesc    objResource = objDesc;
    Indices    indices     = Indices(objResource.indexAddress);
    Vertices   vertices    = Vertices(objResource.vertexAddress);

  //  // Indices of the triangle
    ivec3 ind = indices.i[gl_PrimitiveID].xyz;
  
  //  // Vertex of the triangle
    Vertex v0 = vertices.v[ind.x];
    Vertex v1 = vertices.v[ind.y];
    Vertex v2 = vertices.v[ind.z];

    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    const vec3 normal = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;
    //const vec3 pos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;

    //RandomRay ranRay = rR.i[0];
    vec3  origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    //vec3 origin = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0f));
    vec3 worldNormal = normalize(vec3(transpose(inverse(mat3(trans))) * normal));

    /*payload.hitValue = worldNormal;

    return;*/

    vec3 sum = vec3(0.0);
    
    Payload tempPayload = payload;

    uvec2 pixelCoord = gl_LaunchIDEXT.xy;
    uvec2 screenSize = gl_LaunchSizeEXT.xy;
    uint rngState = pixelCoord.y * screenSize.x + pixelCoord.x;

    for(int i = 0; i < RANDOMNUM; i++){
        float tMin   = 0.0;
        float tMax   = 100.0;

        payload.depth = tempPayload.depth - 1;

        /*vec3  rayDir = ranRay.randomRay[i];
        if (dot(rayDir, normal) < 0.0)
        {
            rayDir = -rayDir;
        }*/

        //vec3  rayDir = rotation * ranRay.randomRay[i];
        vec3 rayDir = RandomHemisphereDirection(worldNormal, rngState);

        uint  flags  = gl_RayFlagsOpaqueEXT;//gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
        traceRayEXT(topLevelAS,  // acceleration structure
                    flags,       // rayFlags
                    0xFF,        // cullMask
                    0,           // sbtRecordOffset
                    1,           // sbtRecordStride
                    1,           // missIndex
                    origin + 0.001 * rayDir,      // ray origin
                    tMin,        // ray min range
                    rayDir,      // ray direction
                    tMax,        // ray max range
                    0            // payload (location = 0)
        );
        
        sum += payload.hitValue * max(dot(worldNormal, rayDir), 0.0);
    }

    //payload.depth = tempPayload.depth;

    payload.hitValue = 2.0 * sum * color / float(RANDOMNUM);
    
    if(isLight == TRUE){
        payload.hitValue += vec3(1.0) * 100;//vec3(1.0);
    }
})";

/*
const char* chit_src2 = R"(
)";
*/


const char* miss_src2 = R"(
#version 460
#extension GL_EXT_ray_tracing : enable
//#extension GL_EXT_nonuniform_qualifier : enable
//#extension GL_EXT_scalar_block_layout : enable
//#extension GL_GOOGLE_include_directive : enable
//
//#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
//#extension GL_EXT_buffer_reference2 : require

//#define RANDOMNUM 500

struct Payload
{
    uint depth;
    vec3 hitValue;
};

layout(location = 0) rayPayloadInEXT Payload payload;

void main()
{
    payload.hitValue = vec3(0.5);
})";

#pragma endregion

struct Model
{
    /*struct alignas(16) Vertex {
        alignas(16) float3 pos;
        alignas(16) float3 normal;
        alignas(8)  float2 tex;
    };*/

    struct alignas(16) Vertex {
        float3 pos;
        float padding1;
        float3 normal;
        float padding2;
        float2 tex;
        float2 padding3;
    };

    vector<float3> mPositions;
    vector<Vertex> mVertices;
    vector<unsigned int> mIndices;

    IBuffer _verticesBuffer;
    IBuffer _indicesBuffer;

    void load(const std::string& filePath)
    {
        tinyobj::attrib_t attribs;
        vector<tinyobj::shape_t> shapes;
        vector<tinyobj::material_t> materials;

        std::string warn;
        std::string err;
        if (!tinyobj::LoadObj(&attribs, &shapes, &materials, &warn, &err, filePath.c_str()))
            std::cout << err << std::endl;

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{ float3(0, 0, 0), 0, float3(0, 0, 0), 0, float2(0, 0), float2(0, 0) }; // { float3(0, 0, 0), float3(0, 0, 0), float2(0, 0) };

                if (index.vertex_index >= 0) {
                    vertex.pos._x = attribs.vertices[3 * index.vertex_index];
                    vertex.pos._y = attribs.vertices[3 * index.vertex_index + 1];
                    vertex.pos._z = attribs.vertices[3 * index.vertex_index + 2];
                }
                if (index.normal_index >= 0) {
                    vertex.normal._x = attribs.normals[3 * index.normal_index];
                    vertex.normal._y = attribs.normals[3 * index.normal_index + 1];
                    vertex.normal._z = attribs.normals[3 * index.normal_index + 2];
                }
                if (index.texcoord_index >= 0) {
                    // teapot model의 경우 min = 0, max = 2 이므로 임시로 보간
                    vertex.tex._x = attribs.texcoords[2 * index.texcoord_index] / 2.0f;
                    vertex.tex._y = attribs.texcoords[2 * index.texcoord_index + 1] / 2.0f;
                }

                mPositions.push_back(vertex.pos);
                mVertices.push_back(vertex);
                mIndices.push_back(static_cast<uint32_t>(mIndices.size()));
                //mIndices.push_back(static_cast<uint32_t>(index.vertex_index));
            }
        }

        _verticesBuffer = std::move(createBuffer(
            mVertices.data(), sizeof(mVertices[0]) * mVertices.size(), DescriptorType::STORAGE_BUFFER,
            static_cast<uint32>(BufferUsageFlagBits::BUFFER_USAGE_STORAGE_BUFFER_BIT) | static_cast<uint32>(BufferUsageFlagBits::BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT),
            static_cast<uint32>(MemoryPropertyFlagBits::MEMORY_PROPERTY_HOST_VISIBLE_BIT) | static_cast<uint32>(MemoryPropertyFlagBits::MEMORY_PROPERTY_HOST_COHERENT_BIT),
            "VertexBuffer For Model"));

        _indicesBuffer = std::move(createBuffer(
            mIndices.data(), sizeof(mIndices[0]) * mIndices.size(), DescriptorType::STORAGE_BUFFER,
            static_cast<uint32>(BufferUsageFlagBits::BUFFER_USAGE_STORAGE_BUFFER_BIT) | static_cast<uint32>(BufferUsageFlagBits::BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT),
            static_cast<uint32>(MemoryPropertyFlagBits::MEMORY_PROPERTY_HOST_VISIBLE_BIT) | static_cast<uint32>(MemoryPropertyFlagBits::MEMORY_PROPERTY_HOST_COHERENT_BIT),
            "IndexBuffer For Model"));
    }
};

struct SceneConstantData {
    float cameraPos[3];
    float yFov_degree;
    float _time;
} dataSrc;

float cameraForward[3];
float cameraRight[3];
float cameraUp[3];

struct ObjDesc
{
    //int      txtOffset;             // Texture index offset in the array of textures
    uint64_t vertexAddress;         // Address of the Vertex buffer
    uint64_t indexAddress;          // Address of the index buffer
    //uint64_t materialAddress;       // Address of the material buffer
    //uint64_t materialIndexAddress;  // Address of the triangle material index buffer
};

struct RandomRay
{
    float3 randomRay[RANDOMNUM] = { float3(0, 0, 0), };
    float padding = 0;
} rR;

class RaytracingRenderer
{
    static IPipeline _raytracingPipeline;

    static Model _model;
    static Model _sphere;
    static Model _plane;

    static AccelerationStructure _tlas;
    static IImage _outImage;
    static IBuffer _sceneConstantBuffer;

    static IBuffer _objBuffer;

    static IBuffer _randomRayBuffer;

public:
    static void initialize()
    {
        initVulkan(WIDTH, HEIGHT, window);

        /////////// blas & tlas
        vector<CreateBlasData> blasData;

        _model.load("../bunny.obj");

        static float4x4 blasTransformArrBunny[] =
        {
            {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                1.5f, 0.5f, 0, 0
            },
            {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                -1.5f, 0.5f, 0, 0
            },
            {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                -1.5f, -2.9f, 0, 0
            },
            {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                1.5f, -2.9f, 0, 0
            },
        };

        for (uint32 i = 0; i < 4; ++i)
        {
            blasData.push_back(CreateBlasData{
                    _model.mPositions.data(), sizeof(_model.mPositions[0]), static_cast<uint32>(_model.mPositions.size()),
                    _model.mIndices.data(), sizeof(_model.mIndices[0]), static_cast<uint32>(_model.mIndices.size()),
                    blasTransformArrBunny[i]
                });
        }

        _sphere.load("../sphere.obj");

        static float4x4 blasTransformArrSphere[] =
        {
            /*{
                0.2f, 0, 0, 0,
                0, 0.2f, 0, 0,
                0, 0, 0.2f, 0,
                0, 2.8f, 0, 0
            },*/
            {
                0.5f, 0, 0, 0,
                0, 0.5f, 0, 0,
                0, 0, 0.5f, 0,
                0, 2.0f, 0, 0
            },
        };

        blasData.push_back(CreateBlasData{
                _sphere.mPositions.data(), sizeof(_sphere.mPositions[0]), static_cast<uint32>(_sphere.mPositions.size()),
                _sphere.mIndices.data(), sizeof(_sphere.mIndices[0]), static_cast<uint32>(_sphere.mIndices.size()),
                blasTransformArrSphere[0]
            });

        _plane.load("../plane.obj");

        static float4x4 blasTransformArrPlane[] =
        {
            {                   // Back
                3, 0, 0, 0,
                0, 3, 0, 0,
                0, 0, 3, 0,
                0, 0, -3, 0
            },
            {                   // Top
                -3, 0, 0, 0,
                0, 0, 3, 0,
                0, -3, 0, 0,
                0, 3, 0, 0
            },
            {                   // Bottom
                3, 0, 0, 0,
                0, 0, -3, 0,
                0, 3, 0, 0,
                0, -3, 0, 0
            },
            {                   // Right
                0, 0, 3, 0,
                0, 3, 0, 0,
                -3, 0, 0, 0,
                3, 0, 0, 0
            },
            {                   // Left
                0, 0, -3, 0,
                0, 3, 0, 0,
                3, 0, 0, 0,
                -3, 0, 0, 0
            },
        };

        for (uint32 i = 0; i < 5; ++i)
        {
            blasData.push_back(CreateBlasData{
                    _plane.mPositions.data(), sizeof(_plane.mPositions[0]), static_cast<uint32>(_plane.mPositions.size()),
                    _plane.mIndices.data(), sizeof(_plane.mIndices[0]), static_cast<uint32>(_plane.mIndices.size()),
                    blasTransformArrPlane[i]
                });
        }

        static float4x4 tlasTransformArr[] =
        {
            {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 0
            },
        };
        vector<CreateTLASData> dataArr;
        for (uint32 i = 0; i < 1; ++i)
            dataArr.push_back(CreateTLASData{ blasData , 100, 0, tlasTransformArr[i] });

        _tlas = std::move(createTLAS(dataArr));

        // initialize vulkan lib
        struct HitgCustomData {
            float color[3];
            uint isLight;          // TRUE or FALSE
            ObjDesc objDesc;
            float4x4 trans;
        };

        RaytracingPipelineDescription desc;
        desc._raygenSource.push_back(raygen_src);
        desc._missSource.push_back(miss_src1);
        desc._missSource.push_back(miss_src2);
        desc._chitSource.push_back(chit_src1);
        //desc._chitSource.push_back(chit_src2);

        desc._maxPipelineRayRecursionDepth = 5;

        desc._raygenShaderBindingTable._sourceIndexArr.push_back(0);

        desc._missShaderBindingTable._sourceIndexArr.push_back(0);
        desc._missShaderBindingTable._sourceIndexArr.push_back(1);

        desc._chitShaderBindingTable._customDataStride = sizeof(HitgCustomData);// +sizeof(ObjDesc);

        ObjDesc objBunny = ObjDesc{
            //0,
            (uint64_t)_model._verticesBuffer.getBufferAddress(),
            (uint64_t)_model._indicesBuffer.getBufferAddress()
        };

        ObjDesc objSphere = ObjDesc{
            //0,
            (uint64_t)_sphere._verticesBuffer.getBufferAddress(),
            (uint64_t)_sphere._indicesBuffer.getBufferAddress()
        };

        ObjDesc objPlane = ObjDesc{
            //0,
            (uint64_t)_plane._verticesBuffer.getBufferAddress(),
            (uint64_t)_plane._indicesBuffer.getBufferAddress()
        };

        HitgCustomData data1{ 1.0f, 1.0f, 1.0f, FALSE, objBunny , blasTransformArrBunny[0] };
        desc._chitShaderBindingTable._sourceIndexArr.push_back(0);
        desc._chitShaderBindingTable._customDataArr.push_back(&data1);
        HitgCustomData data2{ 1.0f, 1.0f, 1.0f, FALSE, objBunny , blasTransformArrBunny[1] };
        desc._chitShaderBindingTable._sourceIndexArr.push_back(0);
        desc._chitShaderBindingTable._customDataArr.push_back(&data2);
        HitgCustomData data3{ 1.0f, 1.0f, 1.0f, FALSE, objBunny , blasTransformArrBunny[2] };
        desc._chitShaderBindingTable._sourceIndexArr.push_back(0);
        desc._chitShaderBindingTable._customDataArr.push_back(&data3);
        HitgCustomData data4{ 1.0f, 1.0f, 1.0f, FALSE, objBunny , blasTransformArrBunny[3] };
        desc._chitShaderBindingTable._sourceIndexArr.push_back(0);
        desc._chitShaderBindingTable._customDataArr.push_back(&data4);
        HitgCustomData data5{ 1.0f, 1.0f, 1.0f, TRUE, objSphere , blasTransformArrSphere[0]};
        desc._chitShaderBindingTable._sourceIndexArr.push_back(0);
        desc._chitShaderBindingTable._customDataArr.push_back(&data5);
        HitgCustomData data6{ 1.0f, 0.0f, 0.0f, FALSE, objPlane , blasTransformArrPlane[0] };
        desc._chitShaderBindingTable._sourceIndexArr.push_back(0);
        desc._chitShaderBindingTable._customDataArr.push_back(&data6);
        HitgCustomData data7{ 0.0f, 1.0f, 0.0f, FALSE, objPlane , blasTransformArrPlane[1] };
        desc._chitShaderBindingTable._sourceIndexArr.push_back(0);
        desc._chitShaderBindingTable._customDataArr.push_back(&data7);
        HitgCustomData data8{ 0.0f, 0.0f, 1.0f, FALSE, objPlane , blasTransformArrPlane[2] };
        desc._chitShaderBindingTable._sourceIndexArr.push_back(0);
        desc._chitShaderBindingTable._customDataArr.push_back(&data8);
        HitgCustomData data9{ 1.0f, 1.0f, 0.0f, FALSE, objPlane , blasTransformArrPlane[3] };
        desc._chitShaderBindingTable._sourceIndexArr.push_back(0);
        desc._chitShaderBindingTable._customDataArr.push_back(&data9);
        HitgCustomData data0{ 1.0f, 0.0f, 1.0f, FALSE, objPlane , blasTransformArrPlane[4] };
        desc._chitShaderBindingTable._sourceIndexArr.push_back(0);
        desc._chitShaderBindingTable._customDataArr.push_back(&data0);

        _raytracingPipeline = std::move(createRayTracingPipeline(desc));

        _outImage = std::move(createImage());
        _sceneConstantBuffer = std::move(createBuffer(
            nullptr, sizeof(SceneConstantData), DescriptorType::UNIFORM_BUFFER,
            static_cast<uint32>(BufferUsageFlagBits::BUFFER_USAGE_UNIFORM_BUFFER_BIT),
            static_cast<uint32>(MemoryPropertyFlagBits::MEMORY_PROPERTY_HOST_VISIBLE_BIT) | static_cast<uint32>(MemoryPropertyFlagBits::MEMORY_PROPERTY_HOST_COHERENT_BIT),
            "_sceneConstantBuffer"));

        /*
        _randomRayBuffer = std::move(createBuffer(
            nullptr, sizeof(RandomRay), DescriptorType::STORAGE_BUFFER,
            static_cast<uint32>(BufferUsageFlagBits::BUFFER_USAGE_STORAGE_BUFFER_BIT),
            static_cast<uint32>(MemoryPropertyFlagBits::MEMORY_PROPERTY_HOST_VISIBLE_BIT) | static_cast<uint32>(MemoryPropertyFlagBits::MEMORY_PROPERTY_HOST_COHERENT_BIT),
            "_randomRayBuffer"));

        for (int i = 0; i < RANDOMNUM; i++) {
            float3 temp = sample_upper_sphere_HG();
            rR.randomRay[i] = temp;
        }

        _randomRayBuffer.upload(&rR, sizeof(RandomRay));
        */
    }

    static void update()
    {
        // update uniform buffer
        //dataSrc = { 0, 0, 5, 60, GetTickCount64() / 1000.0f };
        _sceneConstantBuffer.upload(&dataSrc, sizeof(SceneConstantData));

    }

    static void render()
    {
        bindPipeline(_raytracingPipeline, _outImage);
        {
            bindBuffer("topLevelAS", &_tlas);
            bindBuffer("outImage", &_outImage);
            bindBuffer("g", &_sceneConstantBuffer);
            //bindBuffer("rR", &_randomRayBuffer);

            traceRays();
        }
        unbindPipeline();
    }
};
IPipeline RaytracingRenderer::_raytracingPipeline;
Model RaytracingRenderer::_model;
Model RaytracingRenderer::_sphere;
Model RaytracingRenderer::_plane;
AccelerationStructure RaytracingRenderer::_tlas;
IImage RaytracingRenderer::_outImage;
IBuffer RaytracingRenderer::_sceneConstantBuffer;
IBuffer RaytracingRenderer::_objBuffer;
IBuffer RaytracingRenderer::_randomRayBuffer;

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}
const bool initWindow()
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return false;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    if (!glfwVulkanSupported())
    {
        printf("GLFW: Vulkan Not Supported\n");
        return false;
    }

    // Callbacks
    glfwSetCursorPosCallback(window, mouseMove);
    glfwSetKeyCallback(window, keyboard);



    return true;
}

void mainLoop()
{
    ImGuiIO& io = ImGui::GetIO();

    bool show_another_window = false;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Resize swap chain?
        //int fb_width, fb_height;
        //glfwGetFramebufferSize(window, &fb_width, &fb_height);
        //if (fb_width > 0 && fb_height > 0 && (g_SwapChainRebuild || g_MainWindowData.Width != fb_width || g_MainWindowData.Height != fb_height))
        //{
        //    ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
        //    ImGui_ImplVulkanH_CreateOrResizeWindow(g_Instance, g_PhysicalDevice, g_Device, &g_MainWindowData, g_QueueFamily, g_Allocator, fb_width, fb_height, g_MinImageCount);
        //    g_MainWindowData.FrameIndex = 0;
        //    g_SwapChainRebuild = false;
        //}
        //if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
        //{
        //    ImGui_ImplGlfw_Sleep(10);
        //    continue;
        //}

        // Start the Dear ImGui frame
        startFrame();

        // update
        RaytracingRenderer::update();

        // render
        RaytracingRenderer::render();

        // ui
        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
        bindRenderPass();
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f

            if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }
        // 3. Show another simple window.
        if (true)
        {
            ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }
        unbindRenderPass();

        endFrame();
    }
}

void cleanup()
{
    destroyVulkan();

    glfwDestroyWindow(window);
    glfwTerminate();
}

void run() {
    // engine initialize
    const bool successInitGLFW = initWindow();
    if (successInitGLFW == false)
        return;

    RaytracingRenderer::initialize();

    dataSrc = { 0, 0, 8.2f, 60, GetTickCount64() / 1000.0f };
    cameraForward[0] = 0;
    cameraForward[1] = 0;
    cameraForward[2] = 1.0f;
    cameraRight[0] = 1.0f;
    cameraRight[1] = 0;
    cameraRight[2] = 0;
    cameraUp[0] = 0;
    cameraUp[1] = 1.0f;
    cameraUp[2] = 0;


    mainLoop();

    cleanup();
}

int main()
{

    try {
        run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

float bias = 0.1f;

void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    switch (key) {
        //case GLFW_KEY_Q:
    case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
    case GLFW_KEY_W:
        dataSrc.cameraPos[0] -= bias * cameraForward[0];
        dataSrc.cameraPos[1] -= bias * cameraForward[1];
        dataSrc.cameraPos[2] -= bias * cameraForward[2];
        break;
    case GLFW_KEY_S:
        dataSrc.cameraPos[0] -= -bias * cameraForward[0];
        dataSrc.cameraPos[1] -= -bias * cameraForward[1];
        dataSrc.cameraPos[2] -= -bias * cameraForward[2];
        break;
    case GLFW_KEY_D:
        dataSrc.cameraPos[0] += bias * cameraRight[0];
        dataSrc.cameraPos[1] += bias * cameraRight[1];
        dataSrc.cameraPos[2] += bias * cameraRight[2];
        break;
    case GLFW_KEY_A:
        dataSrc.cameraPos[0] += -bias * cameraRight[0];
        dataSrc.cameraPos[1] += -bias * cameraRight[1];
        dataSrc.cameraPos[2] += -bias * cameraRight[2];
        break;
    case GLFW_KEY_E:
        dataSrc.cameraPos[0] += bias * cameraUp[0];
        dataSrc.cameraPos[1] += bias * cameraUp[1];
        dataSrc.cameraPos[2] += bias * cameraUp[2];
        break;
    case GLFW_KEY_Q:
        dataSrc.cameraPos[0] += -bias * cameraUp[0];
        dataSrc.cameraPos[1] += -bias * cameraUp[1];
        dataSrc.cameraPos[2] += -bias * cameraUp[2];
        break;
    }
}

void mouseMove(GLFWwindow* window, double xs, double ys)
{
    //if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
    //    return;
    //double x = xs * dpiScaling; // for HiDPI
    //double y = ys * dpiScaling; // for HiDPI
    //select(window, x, y);
}