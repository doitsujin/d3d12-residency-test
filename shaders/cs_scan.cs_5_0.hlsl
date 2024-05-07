globallycoherent RWStructuredBuffer<uint> buf : register(u0);

[numthreads(256, 1, 1)]
void main(in uint3 tid : SV_DispatchThreadID) {
  /* always false in practice! */
  if (buf[tid.x] == 0xdeadbeef)
    buf[tid.x] = 0;
}
