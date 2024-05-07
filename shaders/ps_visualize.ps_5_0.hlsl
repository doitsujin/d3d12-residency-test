cbuffer DrawInfo : register(b0) {
  uint color;
  float intensity;
};

float4 main() : SV_TARGET {
  float4 result;
  result.xyz = intensity * float3(
    (color >>  0) & 0xff,
    (color >>  8) & 0xff,
    (color >> 16) & 0xff) / 255.0f;
  result.w = 1.0f;
  return result;
}
