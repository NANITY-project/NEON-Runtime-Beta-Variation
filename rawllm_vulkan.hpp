#pragma once
// =============================================================================
// rawllm_vulkan.hpp — EXPERIMENTAL Vulkan compute backend for the F32 matvec
// path (proj_all_positions()'s dequantize_row()+dot_f32() loop).
//
// STATUS: experimental / not wired into the engine's hot path. This header
// is usable stand-alone (construct a VulkanMatvecBackend, call matvec()) but
// rawllm_forward.hpp does not call into it yet — CPU SIMD (rawllm_simd_
// dispatch.hpp) remains the only path actually exercised by NEON-3.cpp.
// Wiring this in as a real dispatch target (deciding when GPU beats CPU for
// a given tensor size, handling the upload/readback cost, falling back
// cleanly when no Vulkan device is present) is future work — this is
// intentionally scoped to "does the GPU kernel produce correct results",
// not "should the engine use it by default".
//
// Only compiled in when USE_VULKAN is defined (NEON.py's Vulkan build
// toggle). Requires the Vulkan SDK loader + validation-free headers
// (vulkan/vulkan.h) and a precompiled shaders/matvec_f32.spv — this repo
// commits the GLSL source (shaders/matvec_f32.comp) and expects the build
// step to invoke `glslc shaders/matvec_f32.comp -o shaders/matvec_f32.spv`,
// the same way NEON.py already shells out to the platform compiler for the
// CPU intrinsics flags; no compiled shader binary is committed here.
//
// Scope: F32 matvec only, matching matvec_f32.comp. Quantized (Q4_0/Q8_0)
// fused GPU kernels are a documented follow-up once this lands, is
// validated against the CPU dot_f32() path on real hardware (this sandbox
// has no GPU — see the correctness note in the constructor doc below), and
// the upload/readback cost model is understood well enough to know when
// GPU dispatch actually wins over the CPU thread pool.
// =============================================================================

#if defined(USE_VULKAN)

#include "rawllm_common.hpp"
#include <vulkan/vulkan.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>

namespace vk_backend {

inline std::vector<uint32_t> read_spirv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("rawllm_vulkan: couldn't open SPIR-V file: " + path +
        " (did the build step run `glslc shaders/matvec_f32.comp -o " + path + "`?)");
    size_t size = (size_t)f.tellg();
    if (size % 4 != 0)
        throw std::runtime_error("rawllm_vulkan: SPIR-V file size not a multiple of 4: " + path);
    std::vector<uint32_t> buf(size / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)size);
    return buf;
}

inline void vk_check(VkResult r, const char* what) {
    if (r != VK_SUCCESS)
        throw std::runtime_error(std::string("rawllm_vulkan: ") + what + " failed (VkResult=" + std::to_string((int)r) + ")");
}

// Experimental single-device, single-queue Vulkan compute backend for the
// F32 matvec kernel. One instance owns its own VkInstance/VkDevice — this
// is NOT meant to be constructed per-call; construct once, call matvec()
// repeatedly (mirrors util::ThreadPool's lifecycle for the CPU path).
//
// CORRECTNESS NOTE: this class has been compile-tested (SPIR-V codegen,
// pipeline/descriptor-set-layout construction, push-constant layout
// matching matvec_f32.comp's `Params` block) but NOT run against a
// physical or software Vulkan device — this sandbox has no GPU and no
// swiftshader/lavapipe fallback installed. Treat the numerical output as
// unverified until it has been run and diffed against dot_f32() on real
// hardware; that verification is the first item of the documented
// follow-up work, not something this header can self-certify.
class VulkanMatvecBackend {
public:
    explicit VulkanMatvecBackend(const std::string& spirv_path = "shaders/matvec_f32.spv") {
        create_instance();
        pick_physical_device();
        create_device_and_queue();
        create_descriptor_and_pipeline(spirv_path);
    }

    ~VulkanMatvecBackend() {
        if (device_ != VK_NULL_HANDLE) {
            if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
            if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
            if (desc_set_layout_) vkDestroyDescriptorSetLayout(device_, desc_set_layout_, nullptr);
            if (desc_pool_) vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
            if (cmd_pool_) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
    }

    VulkanMatvecBackend(const VulkanMatvecBackend&) = delete;
    VulkanMatvecBackend& operator=(const VulkanMatvecBackend&) = delete;

    // Computes y[r] = dot(W[r*cols : r*cols+cols], x[0:cols]) for r in
    // [0, rows). W is row-major, already-dequantized F32 (cols*rows
    // floats); x is one shared activation vector (cols floats); y is
    // caller-owned output (rows floats). Mirrors proj_all_positions()'s
    // per-tensor F32 matvec exactly, just executed on the GPU instead of
    // the CPU thread pool.
    void matvec(const float* W, const float* x, float* y, uint32_t cols, uint32_t rows) {
        VkDeviceSize w_bytes = (VkDeviceSize)cols * rows * sizeof(float);
        VkDeviceSize x_bytes = (VkDeviceSize)cols * sizeof(float);
        VkDeviceSize y_bytes = (VkDeviceSize)rows * sizeof(float);

        Buffer w_buf = make_buffer(w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer x_buf = make_buffer(x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer y_buf = make_buffer(y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        upload(w_buf, W, w_bytes);
        upload(x_buf, x, x_bytes);

        VkDescriptorSet dset = allocate_and_bind_descriptor_set(w_buf.buf, x_buf.buf, y_buf.buf);

        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmd_pool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vk_check(vkAllocateCommandBuffers(device_, &cbai, &cmd), "vkAllocateCommandBuffers");

        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(cmd, &cbbi), "vkBeginCommandBuffer");

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &dset, 0, nullptr);
        struct { uint32_t cols, rows; } push{cols, rows};
        vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        // One workgroup per output row, matching matvec_f32.comp's
        // `gl_WorkGroupID.x == row` indexing.
        vkCmdDispatch(cmd, rows, 1, 1);

        vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vk_check(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
        vk_check(vkQueueWaitIdle(queue_), "vkQueueWaitIdle");

        vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);

        download(y_buf, y, y_bytes);

        destroy_buffer(w_buf);
        destroy_buffer(x_buf);
        destroy_buffer(y_buf);
    }

private:
    struct Buffer { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkDeviceSize size = 0; };

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    VkDescriptorSetLayout desc_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;

    void create_instance() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "NEON-Vulkan-Matvec";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        vk_check(vkCreateInstance(&ici, nullptr, &instance_), "vkCreateInstance");
    }

    void pick_physical_device() {
        uint32_t count = 0;
        vk_check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices");
        if (count == 0) throw std::runtime_error("rawllm_vulkan: no Vulkan-capable device found");
        std::vector<VkPhysicalDevice> devices(count);
        vk_check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "vkEnumeratePhysicalDevices");
        // First device with a compute-capable queue family wins -- no
        // discrete-vs-integrated preference logic yet (documented follow-up:
        // prefer discrete GPUs, fall back to integrated, surface a clear
        // error when only a CPU/software rasterizer ICD is present).
        for (auto d : devices) {
            uint32_t qcount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qcount, nullptr);
            std::vector<VkQueueFamilyProperties> qprops(qcount);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qcount, qprops.data());
            for (uint32_t i = 0; i < qcount; ++i) {
                if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    phys_ = d;
                    queue_family_ = i;
                    return;
                }
            }
        }
        throw std::runtime_error("rawllm_vulkan: no device with a compute queue family found");
    }

    void create_device_and_queue() {
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = queue_family_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        vk_check(vkCreateDevice(phys_, &dci, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = queue_family_;
        vk_check(vkCreateCommandPool(device_, &cpci, nullptr, &cmd_pool_), "vkCreateCommandPool");
    }

    void create_descriptor_and_pipeline(const std::string& spirv_path) {
        VkDescriptorSetLayoutBinding bindings[3]{};
        for (int i = 0; i < 3; ++i) {
            bindings[i].binding = (uint32_t)i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.bindingCount = 3;
        dslci.pBindings = bindings;
        vk_check(vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &desc_set_layout_), "vkCreateDescriptorSetLayout");

        VkDescriptorPoolSize psize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &psize;
        dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        vk_check(vkCreateDescriptorPool(device_, &dpci, nullptr, &desc_pool_), "vkCreateDescriptorPool");

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &desc_set_layout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &pipeline_layout_), "vkCreatePipelineLayout");

        auto code = read_spirv(spirv_path);
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = code.size() * sizeof(uint32_t);
        smci.pCode = code.data();
        VkShaderModule module;
        vk_check(vkCreateShaderModule(device_, &smci, nullptr, &module), "vkCreateShaderModule");

        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";

        VkComputePipelineCreateInfo cpci2{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci2.stage = stage;
        cpci2.layout = pipeline_layout_;
        vk_check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci2, nullptr, &pipeline_), "vkCreateComputePipelines");

        vkDestroyShaderModule(device_, module, nullptr);
    }

    uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
                return i;
        }
        throw std::runtime_error("rawllm_vulkan: no suitable memory type (host-visible+coherent staging buffer)");
    }

    // All buffers are host-visible + coherent (no staging/transfer-queue
    // split) — the simplest correct thing, not the fastest. A device-local
    // buffer + explicit staging upload is the obvious next optimization
    // once this path is validated and actually wired in.
    Buffer make_buffer(VkDeviceSize size, VkBufferUsageFlags usage) {
        Buffer b; b.size = size;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vk_check(vkCreateBuffer(device_, &bci, nullptr, &b.buf), "vkCreateBuffer");

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, b.buf, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vk_check(vkAllocateMemory(device_, &mai, nullptr, &b.mem), "vkAllocateMemory");
        vk_check(vkBindBufferMemory(device_, b.buf, b.mem, 0), "vkBindBufferMemory");
        return b;
    }

    void destroy_buffer(Buffer& b) {
        if (b.buf) vkDestroyBuffer(device_, b.buf, nullptr);
        if (b.mem) vkFreeMemory(device_, b.mem, nullptr);
        b = Buffer{};
    }

    void upload(Buffer& b, const void* src, VkDeviceSize bytes) {
        void* mapped;
        vk_check(vkMapMemory(device_, b.mem, 0, bytes, 0, &mapped), "vkMapMemory (upload)");
        std::memcpy(mapped, src, (size_t)bytes);
        vkUnmapMemory(device_, b.mem);
    }

    void download(Buffer& b, void* dst, VkDeviceSize bytes) {
        void* mapped;
        vk_check(vkMapMemory(device_, b.mem, 0, bytes, 0, &mapped), "vkMapMemory (download)");
        std::memcpy(dst, mapped, (size_t)bytes);
        vkUnmapMemory(device_, b.mem);
    }

    VkDescriptorSet allocate_and_bind_descriptor_set(VkBuffer w, VkBuffer x, VkBuffer y) {
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = desc_pool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &desc_set_layout_;
        VkDescriptorSet dset;
        vk_check(vkAllocateDescriptorSets(device_, &dsai, &dset), "vkAllocateDescriptorSets");

        VkDescriptorBufferInfo binfos[3] = {
            {w, 0, VK_WHOLE_SIZE}, {x, 0, VK_WHOLE_SIZE}, {y, 0, VK_WHOLE_SIZE}
        };
        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = dset;
            writes[i].dstBinding = (uint32_t)i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &binfos[i];
        }
        vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);
        return dset;
    }
};

} // namespace vk_backend

#endif // USE_VULKAN
