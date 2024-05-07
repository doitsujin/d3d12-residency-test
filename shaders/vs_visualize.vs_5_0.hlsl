float4 main(in uint vid : SV_VERTEXID) : SV_POSITION {
  return float4(
    -1.0f + 4.0f * float(vid & 1),
    -1.0f + 2.0f * float(vid & 2),
    0.0f, 1.0f);
}
