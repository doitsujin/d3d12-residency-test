RWStructuredBuffer<uint> buf : register(u0);

[numthreads(256, 1, 1)]
void main(in uint3 tid : SV_DispatchThreadID) {
  buf[tid.x] = tid.x;
}
