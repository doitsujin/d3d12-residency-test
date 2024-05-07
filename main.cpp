#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl.h>

#include "./shaders/headers/cs_init.h"
#include "./shaders/headers/cs_scan.h"
#include "./shaders/headers/ps_visualize.h"
#include "./shaders/headers/vs_visualize.h"

template <typename T>
using ComPtr = Microsoft::WRL::ComPtr<T>;


struct CmdArgs {
  /** DXGI adapter index to run the test on. */
  uint32_t adapterIndex = 0;

  /** Size of each individual BO, in bytes. */
  uint64_t boSize = 128ull << 20;

  /** Total amount of memory to allocate, in bytes. If 0, the
   *  given percentage will be used instead. */
  uint64_t memSize = 0;
  /** Percentage of available memory to allocate. */
  uint32_t memPercentage = 65;

  /** Amount of memory to scan per frame. Defaults to 1GB. */
  uint64_t scanSize = 1ull << 30;
  /** Number of full scans to perform before closing the application.
   *  Unlimited by default, so the app will run indefinitely. */
  uint32_t scanCount = 0;

  /** Whether to assign unique priorities to each BO. */
  bool prioEnable = false;
  /** Whether to shuffle BO priorities after each full scan. */
  uint32_t prioShuffleInterval = 0;


  bool parseCommandLine() {
    int argc = 0;

    const wchar_t* cmdline = GetCommandLineW();
    wchar_t** argv = CommandLineToArgvW(cmdline, &argc);

    for (int i = 1; i < argc; i++) {
      std::wstring wstr = argv[i];

      if (wstr == L"--adapter-index") {
        int32_t value = -1;

        if (i + 1 < argc)
          value = int32_t(parseNumber(argv[++i]));

        if (value < 0) {
          std::wcerr << "Invalid adapter index" << std::endl;
          return false;
        }

        adapterIndex = value;
      } else if (wstr == L"--bo-size") {
        int64_t value = -1;

        if (i + 1 < argc)
          value = parseMemsize(argv[++i]);

        if (value <= 0 || (value & 0xffff)) {
          std::wcerr << "Invalid BO size" << std::endl;
          return false;
        }

        boSize = value;
      } else if (wstr == L"--mem-size") {
        int64_t size = -1;
        int32_t percentage = -1;

        if (i + 1 < argc) {
          size = parseMemsize(argv[i + 1]);
          percentage = parsePercentage(argv[i + 1]);
          i += 1;
        }

        if (size > 0) {
          memSize = size;
        } else if (percentage > 0) {
          memPercentage = percentage;
        } else {
          std::wcerr << "Invalid memory size" << std::endl;
          return false;
        }
      } else if (wstr == L"--scan-size") {
        int64_t value = -1;

        if (i + 1 < argc)
          value = parseMemsize(argv[++i]);

        if (value <= 0)
          std::wcerr << "Invalid scan size" << std::endl;

        scanSize = value;
      } else if (wstr == L"--scan-count") {
        int32_t value = -1;

        if (i + 1 < argc)
          value = int32_t(parseNumber(argv[++i]));

        if (value < 0) {
          std::wcerr << "Invalid scan count" << std::endl;
          return false;
        }

        scanCount = value;
      } else if (wstr == L"--prio-enable") {
        prioEnable = true;
      } else if (wstr == L"--prio-shuffle") {
        int32_t value = -1;

        if (i + 1 < argc)
          value = int32_t(parseNumber(argv[++i]));

        if (value < 0) {
          std::wcerr << "Invalid shuffle interval" << std::endl;
          return false;
        }

        prioShuffleInterval = value;
      } else {
        std::wcerr << "Unknown argument: " << wstr << std::endl;
        return false;
      }
    }

    LocalFree(argv);
    return true;
  }


  static int64_t parseNumberWithSuffix(const wchar_t* &str) {
    size_t n = 0;
    int64_t v = 0;

    while (str[n] >= L'0' && str[n] <= L'9') {
      v *= 10;
      v += str[n++] - L'0';
    }

    if (!n)
      return -1;

    str += n;
    return v;
  }


  static int64_t parseNumber(const wchar_t* str) {
    int64_t result = parseNumberWithSuffix(str);

    if (str[0] != L'\0')
      return -1;

    return result;
  }


  static int64_t parseMemsize(const wchar_t* str) {
    int64_t result = parseNumberWithSuffix(str);

    if (result < 0 || str[0] == L'\0')
      return result;

    if (str[1] != L'\0')
      return -1;

    switch (str[0]) {
      case L'k': case L'K': return result << 10;
      case L'm': case L'M': return result << 20;
      case L'g': case L'G': return result << 30;
      default: return -1;
    }
  }


  static int64_t parsePercentage(const wchar_t* str) {
    int64_t result = parseNumberWithSuffix(str);
    
    if (str[0] != L'%' || str[1] != L'\0')
      return -1;

    return result;
  }

};


struct Bo {
  ComPtr<ID3D12Heap> heap;
  ComPtr<ID3D12Resource> buffer;
  /* The heap's current residency priority */
  D3D12_RESIDENCY_PRIORITY priority = D3D12_RESIDENCY_PRIORITY_NORMAL;
  /* Measured read bandwidth, in GByte/s */
  double throughput = 0.0;
};


struct VisualizerArgs {
  uint32_t color;
  float intensity;
};


class ResidencyApp {
  static constexpr uint64_t BoDataPerWorkgroup = 1024;
  static constexpr uint64_t BoDataPerDispatch = 32768 * BoDataPerWorkgroup;
public:

  ResidencyApp(HINSTANCE hInstance, HWND hWnd)
  : m_hwnd(hWnd) {
    if (!m_args.parseCommandLine())
      return;

    std::random_device randDev;
    m_rand.seed(randDev());

    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&m_factory));

    if (FAILED(hr)) {
      std::wcerr << "Failed to create DXGI factory, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    if (FAILED(m_factory->EnumAdapters(m_args.adapterIndex, &m_adapter))) {
      if (FAILED(hr = m_factory->EnumAdapters(0, &m_adapter))) {
        std::wcerr << "Failed to enumerate DXGI adapters, hr 0x" << std::hex << hr << "." << std::endl;
        return;
      }

      std::wcerr << "DXGI adatper " << std::dec << m_args.adapterIndex << " not available." << std::endl;
    }

    if (FAILED(hr = D3D12CreateDevice(m_adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)))) {
      std::wcerr << "Failed to create D3D12 device, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc = { };
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    if (FAILED(hr = m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_queue)))) {
      std::wcerr << "Failed to create D3D12 queue, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    if (FAILED(hr = m_queue->GetTimestampFrequency(&m_timerFreq))) {
      std::wcerr << "Failed get GPU timer frequency, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    RECT clientArea = { };
    GetClientRect(m_hwnd, &clientArea);

    DXGI_SWAP_CHAIN_DESC1 swapDesc = { };
    swapDesc.Width = clientArea.right;
    swapDesc.Height = clientArea.bottom;
    swapDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapDesc.BufferCount = 2;
    swapDesc.SampleDesc.Count = 1;
    swapDesc.Scaling = DXGI_SCALING_NONE;
    swapDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapDesc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;

    DXGI_SWAP_CHAIN_FULLSCREEN_DESC swapFsDesc = { };
    swapFsDesc.Windowed = TRUE;

    ComPtr<IDXGISwapChain1> swapchain;

    if (FAILED(hr = m_factory->CreateSwapChainForHwnd(m_queue.Get(),
        m_hwnd, &swapDesc, &swapFsDesc, nullptr, &swapchain))) {
      std::wcerr << "Failed to create swapchain, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    swapchain.As(&m_swapchain);

    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = { };
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.NumDescriptors = swapDesc.BufferCount;

    if (FAILED(hr = m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)))) {
      std::wcerr << "Failed to create RTV descriptor heap, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = { };
    rtvDesc.Format = swapDesc.Format;
    rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;

    for (uint32_t i = 0; i < swapDesc.BufferCount; i++) {
      ComPtr<ID3D12Resource> image;

      if (FAILED(hr = m_swapchain->GetBuffer(i, IID_PPV_ARGS(&image)))) {
        std::wcerr << "Failed to get back buffer, hr 0x" << std::hex << hr << "." << std::endl;
        return;
      }

      m_device->CreateRenderTargetView(image.Get(), &rtvDesc, getRtvHandle(i));
    }

    if (FAILED(hr = m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_cmdPool)))) {
      std::wcerr << "Failed to create command allocator, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    if (FAILED(hr = m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_cmdPool.Get(), nullptr, IID_PPV_ARGS(&m_cmdList)))) {
      std::wcerr << "Failed to create command list, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    if (FAILED(hr = m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) {
      std::wcerr << "Failed to create fence, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    D3D12_ROOT_PARAMETER rootArg = { };
    rootArg.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rootArg.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rootDesc = { };
    rootDesc.NumParameters = 1;
    rootDesc.pParameters = &rootArg;

    if (!createRootSignature(rootDesc, &m_compSig))
      return;

    rootArg.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rootArg.Constants.Num32BitValues = sizeof(VisualizerArgs) / sizeof(uint32_t);
    rootArg.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    if (!createRootSignature(rootDesc, &m_gfxSig))
      return;

    D3D12_COMPUTE_PIPELINE_STATE_DESC csInitDesc = { };
    csInitDesc.pRootSignature = m_compSig.Get();
    csInitDesc.CS = cs_init_dxbc;

    if (FAILED(m_device->CreateComputePipelineState(&csInitDesc, IID_PPV_ARGS(&m_initPso)))) {
      std::wcerr << "Failed to create init pipeline, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC csScanDesc = { };
    csScanDesc.pRootSignature = m_compSig.Get();
    csScanDesc.CS = cs_scan_dxbc;

    if (FAILED(m_device->CreateComputePipelineState(&csScanDesc, IID_PPV_ARGS(&m_scanPso)))) {
      std::wcerr << "Failed to create scan pipeline, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC gfxDesc = { };
    gfxDesc.pRootSignature = m_gfxSig.Get();
    gfxDesc.VS = vs_visualize_dxbc;
    gfxDesc.PS = ps_visualize_dxbc;
    gfxDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
    gfxDesc.SampleMask = 0x1;
    gfxDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    gfxDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    gfxDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    gfxDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    gfxDesc.NumRenderTargets = 1;
    gfxDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    gfxDesc.SampleDesc.Count = 1;

    if (FAILED(m_device->CreateGraphicsPipelineState(&gfxDesc, IID_PPV_ARGS(&m_gfxPso)))) {
      std::wcerr << "Failed to create scan pipeline, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    uint64_t memSize = m_args.memSize;

    if (!memSize) {
      DXGI_ADAPTER_DESC desc = { };
      m_adapter->GetDesc(&desc);

      memSize = (desc.DedicatedVideoMemory * m_args.memPercentage) / 100;

      if (!memSize)
        memSize = m_args.boSize;
    }

    for (size_t i = 0; i * m_args.boSize < memSize; i++) {
      auto& bo = m_bos.emplace_back();

      if (!createBo(bo) || !initBo(bo))
        return;
    }

    m_scanBoCount = (m_args.scanSize + m_args.boSize - 1) / m_args.boSize;

    D3D12_QUERY_HEAP_DESC queryDesc = { };
    queryDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    queryDesc.Count = m_scanBoCount + 1;

    if (FAILED(hr = m_device->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&m_timestampQueries)))) {
      std::wcerr << "Failed to create query heap, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    D3D12_RESOURCE_DESC bufferDesc = { };
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    bufferDesc.Width = sizeof(uint64_t) * queryDesc.Count;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    D3D12_HEAP_PROPERTIES bufferHeap = { };
    bufferHeap.Type = D3D12_HEAP_TYPE_READBACK;

    if (FAILED(hr = m_device->CreateCommittedResource(&bufferHeap, D3D12_HEAP_FLAG_NONE,
        &bufferDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_timestampBuffer)))) {
      std::wcerr << "Failed to create readback buffer, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    if (m_args.prioEnable)
      shufflePriorities();

    m_initialized = true;
  }


  ~ResidencyApp() {

  }


  bool run() {
    if (!m_initialized)
      return false;

    uint32_t lastBo = std::min(m_scanBoIndex + m_scanBoCount, uint32_t(m_bos.size()));

    m_cmdList->SetComputeRootSignature(m_compSig.Get());
    m_cmdList->SetPipelineState(m_scanPso.Get());

    m_cmdList->EndQuery(m_timestampQueries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);

    for (uint32_t i = m_scanBoIndex; i < lastBo; i++) {
      const auto& bo = m_bos.at(i);

      for (uint64_t j = 0; j < m_args.boSize; j += BoDataPerDispatch) {
        uint32_t workgroupCount = uint32_t(std::min(m_args.boSize - j, BoDataPerDispatch) / BoDataPerWorkgroup);

        m_cmdList->SetComputeRootUnorderedAccessView(0, bo.buffer->GetGPUVirtualAddress() + j);
        m_cmdList->Dispatch(workgroupCount, 1, 1);
      }

      m_cmdList->EndQuery(m_timestampQueries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, i - m_scanBoIndex + 1);

      /* Spam barriers to ensure that compute dispatches
       * scanning different BOs do not overlap */
      D3D12_RESOURCE_BARRIER barrier = { };
      barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;

      m_cmdList->ResourceBarrier(1, &barrier);
    }

    m_cmdList->ResolveQueryData(m_timestampQueries.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
      0, lastBo - m_scanBoIndex + 1, m_timestampBuffer.Get(), 0);

    /* Draw render target and present. This is mostly just
     * here to ensure that the app is considered active. */
    static const std::array<float, 4> color = { 1.0f, 1.0f, 1.0f, 0.0f };

    uint32_t imageIndex = m_swapchain->GetCurrentBackBufferIndex();

    ComPtr<ID3D12Resource> swapImage;
    m_swapchain->GetBuffer(imageIndex, IID_PPV_ARGS(&swapImage));

    D3D12_RESOURCE_DESC swapImageDesc = { };
    swapImage->GetDesc(&swapImageDesc);

    D3D12_RESOURCE_BARRIER swapBarrier = { };
    swapBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    swapBarrier.Transition.pResource = swapImage.Get();
    swapBarrier.Transition.Subresource = 0;
    swapBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    swapBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;

    m_cmdList->ResourceBarrier(1, &swapBarrier);

    D3D12_CPU_DESCRIPTOR_HANDLE rtv = getRtvHandle(imageIndex);
    m_cmdList->ClearRenderTargetView(rtv, color.data(), 0, nullptr);
    m_cmdList->OMSetRenderTargets(1, &rtv, false, nullptr);

    m_cmdList->SetGraphicsRootSignature(m_gfxSig.Get());
    m_cmdList->SetPipelineState(m_gfxPso.Get());
    m_cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    constexpr uint32_t QuadSize = 24;
    constexpr uint32_t QuadBorder = 2;

    uint32_t x = 0u;
    uint32_t y = 0u;

    for (const auto& bo : m_bos)
      m_maxThroughput = std::max(m_maxThroughput, bo.throughput);

    for (const auto& bo : m_bos) {
      VisualizerArgs args = { };
      args.intensity = float(std::log2(bo.throughput + 2.0) / std::log2(m_maxThroughput + 2.0));

      if (bo.priority >= D3D12_RESIDENCY_PRIORITY_HIGH)
        args.color = 0x00ff00;
      else if (bo.priority >= D3D12_RESIDENCY_PRIORITY_NORMAL)
        args.color = 0xff0000;
      else
        args.color = 0x0000ff;

      RECT scissor = { };
      scissor.left = x + QuadBorder;
      scissor.top = y + QuadBorder;
      scissor.right = x + QuadSize - QuadBorder;
      scissor.bottom = y + QuadSize - QuadBorder;

      D3D12_VIEWPORT viewport = { };
      viewport.TopLeftX = float(scissor.left);
      viewport.TopLeftY = float(scissor.top);
      viewport.Width = float(scissor.right - scissor.left);
      viewport.Height = float(scissor.bottom - scissor.top);

      m_cmdList->RSSetViewports(1, &viewport);
      m_cmdList->RSSetScissorRects(1, &scissor);
      m_cmdList->SetGraphicsRoot32BitConstants(0, sizeof(args) / sizeof(uint32_t), &args, 0);

      m_cmdList->DrawInstanced(3, 1, 0, 0);

      x += QuadSize;

      if (x + QuadSize > swapImageDesc.Width) {
        x = 0;
        y += QuadSize;
      }
    }

    swapBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    swapBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;

    m_cmdList->ResourceBarrier(1, &swapBarrier);

    /* Submit command list, present and synchronize with the GPU */
    HRESULT hr = m_cmdList->Close();

    if (FAILED(hr)) {
      std::wcerr << "Failed to close command list, hr 0x" << std::hex << hr << "." << std::endl;
      return false;
    }

    ID3D12CommandList* list = m_cmdList.Get();

    m_queue->ExecuteCommandLists(1, &list);
    m_swapchain->Present(1, 0);

    if (!resetCmdList())
      return false;

    /* Read back timestamps */
    void* data = nullptr;

    if (FAILED(hr = m_timestampBuffer->Map(0, nullptr, &data))) {
      std::wcerr << "Failed to map readback buffer, hr 0x" << std::hex << hr << "." << std::endl;
      return false;
    }

    auto timestamps = reinterpret_cast<const uint64_t*>(data);

    for (uint32_t i = 0; i < lastBo - m_scanBoIndex; i++) {
      uint32_t boIndex = i + m_scanBoIndex;
      auto& bo = m_bos.at(boIndex);

      /* We don't need to be super accurate here, just a ballpark estimate
       * of the observed throughput is good enough. If a BO is not resident,
       * we should be limited by PCI-E bandwidth on discrete GPUs. */
      uint64_t delta = timestamps[i + 1] - timestamps[i];
      bo.throughput = double(m_args.boSize) * double(m_timerFreq) / (1000000000.0 * double(delta));
    }

    m_timestampBuffer->Unmap(0, nullptr);

    m_scanBoIndex += m_scanBoCount;

    if (m_scanBoIndex >= m_bos.size()) {
      logScanResults();

      m_scanBoIndex = 0;

      /* Last iteration */
      if (++m_scanNumber == m_args.scanCount)
        return false;

      /* If desired, change BO priorities around every couple of full scans */
      if (m_args.prioEnable && m_args.prioShuffleInterval && !(m_scanNumber % m_args.prioShuffleInterval))
        shufflePriorities();
    }

    return true;
  }

private:

  std::mt19937                      m_rand;

  HWND                              m_hwnd = nullptr;
  CmdArgs                           m_args;

  ComPtr<IDXGIFactory2>             m_factory;
  ComPtr<IDXGIAdapter>              m_adapter;

  ComPtr<ID3D12Device1>             m_device;
  ComPtr<ID3D12CommandQueue>        m_queue;
  uint64_t                          m_timerFreq = 0;

  ComPtr<IDXGISwapChain3>           m_swapchain;
  ComPtr<ID3D12DescriptorHeap>      m_rtvHeap;

  ComPtr<ID3D12CommandAllocator>    m_cmdPool;
  ComPtr<ID3D12GraphicsCommandList> m_cmdList;

  ComPtr<ID3D12RootSignature>       m_compSig;
  ComPtr<ID3D12PipelineState>       m_initPso;
  ComPtr<ID3D12PipelineState>       m_scanPso;

  ComPtr<ID3D12RootSignature>       m_gfxSig;
  ComPtr<ID3D12PipelineState>       m_gfxPso;

  ComPtr<ID3D12Fence>               m_fence;
  uint64_t                          m_fenceValue = 0;

  ComPtr<ID3D12QueryHeap>           m_timestampQueries;
  ComPtr<ID3D12Resource>            m_timestampBuffer;

  std::vector<Bo>                   m_bos;

  uint32_t                          m_scanNumber = 0;
  uint32_t                          m_scanBoIndex = 0;
  uint32_t                          m_scanBoCount = 0;

  bool                              m_initialized = false;

  double                            m_maxThroughput = 0.0;

  D3D12_CPU_DESCRIPTOR_HANDLE getRtvHandle(uint32_t index) const {
    uint32_t increment = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_CPU_DESCRIPTOR_HANDLE handle = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += index * increment;
    return handle;
  }


  bool createBo(Bo& bo) const {
    D3D12_HEAP_DESC heapDesc = { };
    heapDesc.SizeInBytes = m_args.boSize;
    heapDesc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    heapDesc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;

    HRESULT hr = m_device->CreateHeap(&heapDesc, IID_PPV_ARGS(&bo.heap));

    if (FAILED(hr)) {
      std::wcerr << "Failed to create heap, hr 0x" << std::hex << hr << "." << std::endl;
      return false;
    }

    D3D12_RESOURCE_DESC bufferDesc = { };
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Alignment = heapDesc.Alignment;
    bufferDesc.Width = heapDesc.SizeInBytes;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    if (FAILED(hr = m_device->CreatePlacedResource(bo.heap.Get(), 0, &bufferDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&bo.buffer)))) {
      std::wcerr << "Failed to create placed buffer, hr 0x" << std::hex << hr << "." << std::endl;
      return false;
    }

    return true;
  }


  bool initBo(const Bo& bo) {
    m_cmdList->SetComputeRootSignature(m_compSig.Get());
    m_cmdList->SetPipelineState(m_initPso.Get());

    for (uint64_t i = 0; i < m_args.boSize; i += BoDataPerDispatch) {
      uint32_t workgroupCount = uint32_t(std::min(m_args.boSize - i, BoDataPerDispatch) / BoDataPerWorkgroup);

      m_cmdList->SetComputeRootUnorderedAccessView(0, bo.buffer->GetGPUVirtualAddress() + i);
      m_cmdList->Dispatch(workgroupCount, 1, 1);
    }

    D3D12_RESOURCE_BARRIER barrier = { };
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;

    m_cmdList->ResourceBarrier(1, &barrier);

    HRESULT hr = m_cmdList->Close();

    if (FAILED(hr)) {
      std::wcerr << "Failed to close command list, hr 0x" << std::hex << hr << "." << std::endl;
      return false;
    }

    ID3D12CommandList* list = m_cmdList.Get();
    m_queue->ExecuteCommandLists(1, &list);

    return resetCmdList();
  }


  bool resetCmdList() {
    HRESULT hr = m_queue->Signal(m_fence.Get(), ++m_fenceValue);

    if (FAILED(hr)) {
      std::wcerr << "Failed to signal fence, hr 0x" << std::hex << hr << "." << std::endl;
      return false;
    }

    if (FAILED(hr = m_fence->SetEventOnCompletion(m_fenceValue, nullptr))) {
      std::wcerr << "Failed to wait for fence, hr 0x" << std::hex << hr << "." << std::endl;
      return false;
    }

    if (FAILED(hr = m_cmdPool->Reset())) {
      std::wcerr << "Failed to reset command allocator, hr 0x" << std::hex << hr << "." << std::endl;
      return false;
    }

    if (FAILED(hr = m_cmdList->Reset(m_cmdPool.Get(), nullptr))) {
      std::wcerr << "Failed to reset command list, hr 0x" << std::hex << hr << "." << std::endl;
      return false;
    }

    return true;
  }


  void shufflePriorities() {
    static const std::array<D3D12_RESIDENCY_PRIORITY, 3> prioSet = {
      D3D12_RESIDENCY_PRIORITY_LOW,
      D3D12_RESIDENCY_PRIORITY_NORMAL,
      D3D12_RESIDENCY_PRIORITY_HIGH,
    };

    std::vector<D3D12_RESIDENCY_PRIORITY> prios(m_bos.size());
    std::vector<ID3D12Pageable*> bos(m_bos.size());

    for (size_t i = 0; i < bos.size(); i++) {
      prios[i] = prioSet[i % prioSet.size()];
      bos[i] = m_bos[i].heap.Get();
    }

    for (size_t i = 0; i < bos.size(); i++) {
      std::uniform_int_distribution<size_t> dist(i, bos.size() - 1);

      size_t j = dist(m_rand);
      std::swap(prios[i], prios[j]);
    }

    HRESULT hr = m_device->SetResidencyPriority(bos.size(), bos.data(), prios.data());

    if (FAILED(hr)) {
      std::wcerr << "Failed to change BO priorities, hr 0x" << std::hex << hr << "." << std::endl;
      return;
    }

    for (size_t i = 0; i < bos.size(); i++)
      m_bos[i].priority = prios[i];
  }


  void logScanResults() {
    std::wcout << "Scan " << std::dec << m_scanNumber << ": " << std::endl;

    for (size_t i = 0; i < m_bos.size(); i++) {
      if (!(i % 16))
        std::wcout << std::setfill(L' ') << std::setw(4) << i << ": ";

      std::array<std::pair<D3D12_RESIDENCY_PRIORITY, const wchar_t*>, 3> prios = {{
        { D3D12_RESIDENCY_PRIORITY_LOW,     L"[L]", },
        { D3D12_RESIDENCY_PRIORITY_NORMAL,  L"[N]", },
        { D3D12_RESIDENCY_PRIORITY_HIGH,    L"[H]", },
      }};

      const wchar_t* prio = prios[0].second;

      for (size_t j = 1; j < prios.size(); j++) {
        if (m_bos[i].priority >= prios[j].first)
          prio = prios[j].second;
      }

      uint64_t throughput = uint64_t(10.0 * m_bos[i].throughput);
      std::wcout << " " << std::setfill(L' ') << std::setw(4) << (throughput / 10) << "." << std::setw(1) << (throughput % 10) << L" " << prio;

      if (!((i + 1) % 16) || (i + 1 == m_bos.size()))
        std::wcout << std::endl;
    }
  }


  bool createRootSignature(const D3D12_ROOT_SIGNATURE_DESC& desc, ID3D12RootSignature** rootSig) {
    ComPtr<ID3DBlob> rootBlob;

    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rootBlob, nullptr);

    if (FAILED(hr)) {
      std::wcerr << "Failed to serialize root signature, hr 0x" << std::hex << hr << "." << std::endl;
      return false;
    }

    if (FAILED(m_device->CreateRootSignature(0, rootBlob->GetBufferPointer(), rootBlob->GetBufferSize(), IID_PPV_ARGS(rootSig)))) {
      std::wcerr << "Failed to create root signature, hr 0x" << std::hex << hr << "." << std::endl;
      return false;
    }

    return true;
  }

};


LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
  switch (message) {
    case WM_CLOSE:
      PostQuitMessage(0);
      return 0;
  }

  return DefWindowProcW(hWnd, message, wParam, lParam);
}


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
  WNDCLASSEXW wc = { };
  wc.cbSize = sizeof(wc);
  wc.style = CS_HREDRAW | CS_VREDRAW;
  wc.lpfnWndProc = WindowProc;
  wc.hInstance = hInstance;
  wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  wc.hbrBackground = HBRUSH(COLOR_WINDOW);
  wc.lpszClassName = L"WindowClass";
  RegisterClassExW(&wc);

  HWND hWnd = CreateWindowExW(0, L"WindowClass", L"D3D12 residency",
    WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME, 300, 300, 1024, 600,
    nullptr, nullptr, hInstance, nullptr);
  ShowWindow(hWnd, nCmdShow);

  ResidencyApp app(hInstance, hWnd);

  MSG msg;

  while (true) {
    if (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
      
      if (msg.message == WM_QUIT)
        break;
    } else {
      if (!app.run())
        break;
    }
  }

  return msg.wParam;
}
