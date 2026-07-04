/*
 * WiVRn VR streaming
 * Copyright (C) 2026  WiVRn contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

// Minimal raw-Vulkan compute harness used to develop and validate the HEVC
// encoder's compute shaders against the CPU reference. Deliberately tiny and
// dependency-free (just the Vulkan loader) so shaders can be exercised offline;
// the shaders themselves are what eventually run inside WiVRn's vk_bundle.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace wivrn::h267::gpu
{

inline void vkcheck(VkResult r, const char * what)
{
	if (r != VK_SUCCESS)
		throw std::runtime_error(std::string("Vulkan error ") + std::to_string(r) + " at " + what);
}

class vk_compute
{
public:
	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice phys = VK_NULL_HANDLE;
	VkDevice dev = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
	uint32_t qfam = 0;
	VkCommandPool pool = VK_NULL_HANDLE;
	uint32_t host_mem_type = ~0u;      // HOST_VISIBLE|COHERENT — fast CPU writes (uploads)
	uint32_t host_read_mem_type = ~0u; // + HOST_CACHED — fast CPU reads (readback)
	uint32_t gpu_mem_type = ~0u;       // DEVICE_LOCAL(+HOST_VISIBLE via ReBAR) — GPU-only buffers
	// make_buffer kinds: 0 = upload (write-combined, CPU writes), 1 = readback
	// (host-cached, CPU reads), 2 = gpu (device-local; slow uncached CPU access but
	// fastest for GPU-internal buffers — recon/level/scratch never touched by CPU).
	static constexpr int MEM_UPLOAD = 0, MEM_READBACK = 1, MEM_GPU = 2;
	bool owns_device = false;          // true when init() created the device/instance
	float timestamp_period = 1.0f;     // ns per timestamp tick (device limit); for GPU-side profiling
	bool profile_gpu = false;          // when set, begin_batch attaches a timestamp query pool

	struct buffer
	{
		VkBuffer buf = VK_NULL_HANDLE;
		VkDeviceMemory mem = VK_NULL_HANDLE;
		void * ptr = nullptr;
		size_t size = 0;
	};

	struct pipeline
	{
		VkShaderModule shader = VK_NULL_HANDLE;
		VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
		VkPipelineLayout layout = VK_NULL_HANDLE;
		VkPipeline pipe = VK_NULL_HANDLE;
		VkDescriptorPool dpool = VK_NULL_HANDLE;
		VkDescriptorSet dset = VK_NULL_HANDLE;
		int nbindings = 0;
		int n_img = 0; // first n_img bindings are STORAGE_IMAGE, the rest STORAGE_BUFFER
		int push_bytes = 0;
	};

	void init()
	{
		VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
		app.pApplicationName = "wivrn-hevc-compute";
		app.apiVersion = VK_API_VERSION_1_3;
		VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
		ici.pApplicationInfo = &app;
		vkcheck(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");

		uint32_t n = 0;
		vkEnumeratePhysicalDevices(instance, &n, nullptr);
		std::vector<VkPhysicalDevice> devs(n);
		vkEnumeratePhysicalDevices(instance, &n, devs.data());
		if (n == 0)
			throw std::runtime_error("no Vulkan device");
		phys = devs[0];

		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(phys, &props);
		std::fprintf(stderr, "GPU: %s\n", props.deviceName);

		uint32_t qn = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, nullptr);
		std::vector<VkQueueFamilyProperties> qs(qn);
		vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qs.data());
		qfam = ~0u;
		for (uint32_t i = 0; i < qn; ++i)
			if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
			{
				qfam = i;
				break;
			}
		if (qfam == ~0u)
			throw std::runtime_error("no compute queue");

		float prio = 1.0f;
		VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
		qci.queueFamilyIndex = qfam;
		qci.queueCount = 1;
		qci.pQueuePriorities = &prio;
		// Enable features we use in shaders: 8/16-bit storage, int8/int16.
		VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
		f11.storageBuffer16BitAccess = VK_TRUE;
		VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
		f12.pNext = &f11;
		f12.storageBuffer8BitAccess = VK_TRUE;
		f12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
		f12.shaderInt8 = VK_TRUE;
		// Vulkan memory model with device scope: lets shaders use explicit
		// atomic acquire/release (GL_KHR_memory_scope_semantics) for correct
		// cross-workgroup synchronisation (scoreboard reconstruction).
		f12.vulkanMemoryModel = VK_TRUE;
		f12.vulkanMemoryModelDeviceScope = VK_TRUE;
		VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
		f2.pNext = &f12;
		f2.features.shaderInt16 = VK_TRUE;
		f2.features.shaderInt64 = VK_TRUE;
		VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
		dci.pNext = &f2;
		dci.queueCreateInfoCount = 1;
		dci.pQueueCreateInfos = &qci;
		vkcheck(vkCreateDevice(phys, &dci, nullptr, &dev), "vkCreateDevice");
		vkGetDeviceQueue(dev, qfam, 0, &queue);
		owns_device = true;
		finish_setup();
	}

	// Use an already-created device (e.g. WiVRn's shared vk_bundle) instead of
	// creating our own. The caller keeps ownership of the device/queue.
	void adopt(VkPhysicalDevice p, VkDevice d, VkQueue q, uint32_t queue_family)
	{
		phys = p;
		dev = d;
		queue = q;
		qfam = queue_family;
		owns_device = false;
		finish_setup();
	}

	~vk_compute()
	{
		if (pool)
			vkDestroyCommandPool(dev, pool, nullptr);
		if (owns_device)
		{
			if (dev)
				vkDestroyDevice(dev, nullptr);
			if (instance)
				vkDestroyInstance(instance, nullptr);
		}
	}

private:
	void finish_setup()
	{
		VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		pci.queueFamilyIndex = qfam;
		vkcheck(vkCreateCommandPool(dev, &pci, nullptr, &pool), "vkCreateCommandPool");

		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(phys, &props);
		timestamp_period = props.limits.timestampPeriod; // ns/tick, for GPU-stage profiling

		VkPhysicalDeviceMemoryProperties mp;
		vkGetPhysicalDeviceMemoryProperties(phys, &mp);
		const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		const VkMemoryPropertyFlags want_cached = want | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
		// GPU-only buffers want DEVICE_LOCAL + mappable (ReBAR); avoid the AMD
		// device-uncached type so the buffers use the normal L1/L2 cache hierarchy.
		const VkMemoryPropertyFlags want_gpu = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | want;
		const VkMemoryPropertyFlags avoid_gpu = VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
		for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
		{
			auto f = mp.memoryTypes[i].propertyFlags;
			if (host_mem_type == ~0u && (f & want) == want && !(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
				host_mem_type = i;
			// Prefer a coherent+cached type: fast CPU reads with no manual
			// invalidation. On this APU the plain coherent type is write-combined,
			// which reads at ~100 MB/s and dominated the encode time.
			if (host_read_mem_type == ~0u && (f & want_cached) == want_cached && !(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
				host_read_mem_type = i;
			if (gpu_mem_type == ~0u && (f & want_gpu) == want_gpu && !(f & avoid_gpu))
				gpu_mem_type = i;
		}
		if (host_mem_type == ~0u)
			throw std::runtime_error("no host-visible coherent memory");
		if (host_read_mem_type == ~0u)
			host_read_mem_type = host_mem_type; // no cached type; fall back
		if (gpu_mem_type == ~0u)
			gpu_mem_type = host_mem_type; // no mappable device-local type; fall back
	}

public:

	// kind: MEM_UPLOAD (0, write-combined CPU writes), MEM_READBACK (1, host-cached
	// CPU reads), MEM_GPU (2, device-local — GPU-only buffers). Legacy callers pass
	// bool cached which converts to 0/1 unchanged.
	buffer make_buffer(size_t bytes, int kind = MEM_UPLOAD)
	{
		buffer b;
		b.size = bytes;
		VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bci.size = bytes;
		bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		vkcheck(vkCreateBuffer(dev, &bci, nullptr, &b.buf), "vkCreateBuffer");
		VkMemoryRequirements req;
		vkGetBufferMemoryRequirements(dev, b.buf, &req);
		VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		mai.allocationSize = req.size;
		mai.memoryTypeIndex = kind == MEM_GPU ? gpu_mem_type : (kind == MEM_READBACK ? host_read_mem_type : host_mem_type);
		vkcheck(vkAllocateMemory(dev, &mai, nullptr, &b.mem), "vkAllocateMemory");
		vkcheck(vkBindBufferMemory(dev, b.buf, b.mem, 0), "vkBindBufferMemory");
		vkcheck(vkMapMemory(dev, b.mem, 0, bytes, 0, &b.ptr), "vkMapMemory");
		std::memset(b.ptr, 0, bytes);
		return b;
	}

	void destroy_buffer(buffer & b)
	{
		if (b.mem)
			vkUnmapMemory(dev, b.mem), vkFreeMemory(dev, b.mem, nullptr);
		if (b.buf)
			vkDestroyBuffer(dev, b.buf, nullptr);
		b = {};
	}

	// --- Multi-planar YUV 4:2:0 image (G8_B8R8_2PLANE_420) for direct sampling.
	// view_y is an R8 view of plane 0 (luma), view_c an R8G8 view of plane 1
	// (interleaved CbCr, half res). Matches the WiVRn compositor's encoder image,
	// so the recon shader's imageLoad path is exercised by offline tests too. ---
	struct yuv_image
	{
		VkImage img = VK_NULL_HANDLE;
		VkDeviceMemory mem = VK_NULL_HANDLE;
		VkImageView view_y = VK_NULL_HANDLE; // R8, plane 0
		VkImageView view_c = VK_NULL_HANDLE; // R8G8, plane 1
		buffer staging{};                    // host, for CPU uploads (offline)
		int w = 0, h = 0;
	};

	yuv_image make_yuv_image(int w, int h)
	{
		yuv_image y; y.w = w; y.h = h;
		VkFormat plane_fmts[3] = {VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM};
		VkImageFormatListCreateInfo fl{VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
		fl.viewFormatCount = 3; fl.pViewFormats = plane_fmts;
		VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ici.pNext = &fl;
		ici.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
		ici.imageType = VK_IMAGE_TYPE_2D; ici.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		ici.extent = {(uint32_t)w, (uint32_t)h, 1}; ici.mipLevels = 1; ici.arrayLayers = 1;
		ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.tiling = VK_IMAGE_TILING_OPTIMAL;
		ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		vkcheck(vkCreateImage(dev, &ici, nullptr, &y.img), "createYuvImage");
		VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, y.img, &mr);
		VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
		uint32_t mt = ~0u;
		for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
			if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mt = i; break; }
		VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
		vkcheck(vkAllocateMemory(dev, &mai, nullptr, &y.mem), "allocYuvMem");
		vkcheck(vkBindImageMemory(dev, y.img, y.mem, 0), "bindYuvMem");
		auto mkview = [&](VkImageView & v, VkFormat f, VkImageAspectFlags a) {
			VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
			vci.image = y.img; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = f;
			vci.subresourceRange = {a, 0, 1, 0, 1};
			vkcheck(vkCreateImageView(dev, &vci, nullptr, &v), "yuvView");
		};
		mkview(y.view_y, VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_PLANE_0_BIT);
		mkview(y.view_c, VK_FORMAT_R8G8_UNORM, VK_IMAGE_ASPECT_PLANE_1_BIT);
		y.staging = make_buffer((size_t)w * h + (size_t)(w / 2) * (h / 2) * 2);
		return y;
	}

	// Upload extent-sized luma + interleaved CbCr into the image and leave it in
	// GENERAL layout (synchronous; offline path only). Live path samples the
	// compositor image directly and never calls this.
	void upload_yuv(yuv_image & y, const uint8_t * luma, const uint8_t * chroma)
	{
		size_t yb = (size_t)y.w * y.h, cb = (size_t)(y.w / 2) * (y.h / 2) * 2;
		std::memcpy(y.staging.ptr, luma, yb);
		std::memcpy((uint8_t *)y.staging.ptr + yb, chroma, cb);
		VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
		VkCommandBuffer cmd; vkAllocateCommandBuffers(dev, &cbai, &cmd);
		VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd, &bi);
		auto barrier = [&](VkImageLayout o, VkImageLayout n, VkAccessFlags sa, VkAccessFlags da, VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
			VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.oldLayout = o; b.newLayout = n; b.image = y.img;
			b.srcAccessMask = sa; b.dstAccessMask = da; b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
			vkCmdPipelineBarrier(cmd, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
		};
		barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
		VkBufferImageCopy c0{}; c0.bufferOffset = 0; c0.imageSubresource = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1}; c0.imageExtent = {(uint32_t)y.w, (uint32_t)y.h, 1};
		vkCmdCopyBufferToImage(cmd, y.staging.buf, y.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c0);
		VkBufferImageCopy c1{}; c1.bufferOffset = yb; c1.imageSubresource = {VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1}; c1.imageExtent = {(uint32_t)(y.w / 2), (uint32_t)(y.h / 2), 1};
		vkCmdCopyBufferToImage(cmd, y.staging.buf, y.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c1);
		barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		vkEndCommandBuffer(cmd);
		VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; vkCreateFence(dev, &fci, nullptr, &fence);
		VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
		vkQueueSubmit(queue, 1, &si, fence); vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
		vkDestroyFence(dev, fence, nullptr); vkFreeCommandBuffers(dev, pool, 1, &cmd);
	}

	void destroy_yuv_image(yuv_image & y)
	{
		if (y.view_y) vkDestroyImageView(dev, y.view_y, nullptr);
		if (y.view_c) vkDestroyImageView(dev, y.view_c, nullptr);
		if (y.img) vkDestroyImage(dev, y.img, nullptr);
		if (y.mem) vkFreeMemory(dev, y.mem, nullptr);
		destroy_buffer(y.staging);
		y = {};
	}

	void destroy_pipeline(pipeline & p)
	{
		if (p.dpool)
			vkDestroyDescriptorPool(dev, p.dpool, nullptr);
		if (p.pipe)
			vkDestroyPipeline(dev, p.pipe, nullptr);
		if (p.layout)
			vkDestroyPipelineLayout(dev, p.layout, nullptr);
		if (p.dsl)
			vkDestroyDescriptorSetLayout(dev, p.dsl, nullptr);
		if (p.shader)
			vkDestroyShaderModule(dev, p.shader, nullptr);
		p = {};
	}

	static std::vector<uint32_t> read_spv(const char * path)
	{
		FILE * f = std::fopen(path, "rb");
		if (!f)
			throw std::runtime_error(std::string("cannot open spv: ") + path);
		std::fseek(f, 0, SEEK_END);
		long sz = std::ftell(f);
		std::fseek(f, 0, SEEK_SET);
		std::vector<uint32_t> code(sz / 4);
		if (std::fread(code.data(), 1, sz, f) != (size_t)sz)
			throw std::runtime_error("spv read error");
		std::fclose(f);
		return code;
	}

	pipeline make_pipeline(const char * spv_path, int nbindings, int push_bytes = 0, int n_image_bindings = 0)
	{
		auto code = read_spv(spv_path);
		return make_pipeline_from_code(code.data(), code.size(), nbindings, push_bytes, n_image_bindings);
	}

	// Build a compute pipeline from SPIR-V words already in memory (the shape used
	// inside WiVRn, where shaders are embedded as a std::map<string,vector<u32>>).
	// The first n_image_bindings descriptors are STORAGE_IMAGE, the rest STORAGE_BUFFER.
	pipeline make_pipeline_from_code(const uint32_t * code, size_t words, int nbindings, int push_bytes = 0, int n_image_bindings = 0)
	{
		pipeline p;
		p.nbindings = nbindings;
		p.n_img = n_image_bindings;
		p.push_bytes = push_bytes;
		VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
		smci.codeSize = words * 4;
		smci.pCode = code;
		vkcheck(vkCreateShaderModule(dev, &smci, nullptr, &p.shader), "vkCreateShaderModule");

		std::vector<VkDescriptorSetLayoutBinding> binds(nbindings);
		for (int i = 0; i < nbindings; ++i)
			binds[i] = {(uint32_t)i, i < n_image_bindings ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		dlci.bindingCount = nbindings;
		dlci.pBindings = binds.data();
		vkcheck(vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &p.dsl), "descSetLayout");

		VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)push_bytes};
		VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		plci.setLayoutCount = 1;
		plci.pSetLayouts = &p.dsl;
		if (push_bytes > 0)
		{
			plci.pushConstantRangeCount = 1;
			plci.pPushConstantRanges = &pcr;
		}
		vkcheck(vkCreatePipelineLayout(dev, &plci, nullptr, &p.layout), "pipelineLayout");

		VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
		cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		cpci.stage.module = p.shader;
		cpci.stage.pName = "main";
		cpci.layout = p.layout;
		vkcheck(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipe), "computePipeline");

		VkDescriptorPoolSize ps[2];
		uint32_t nps = 0;
		if (nbindings - n_image_bindings > 0)
			ps[nps++] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)(nbindings - n_image_bindings)};
		if (n_image_bindings > 0)
			ps[nps++] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, (uint32_t)n_image_bindings};
		VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		dpci.maxSets = 1;
		dpci.poolSizeCount = nps;
		dpci.pPoolSizes = ps;
		vkcheck(vkCreateDescriptorPool(dev, &dpci, nullptr, &p.dpool), "descPool");
		VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		dsai.descriptorPool = p.dpool;
		dsai.descriptorSetCount = 1;
		dsai.pSetLayouts = &p.dsl;
		vkcheck(vkAllocateDescriptorSets(dev, &dsai, &p.dset), "allocDescSet");
		return p;
	}

	// A single dispatch step of a wavefront: its group count and push constants.
	struct step
	{
		uint32_t gx, gy, gz;
		std::vector<uint8_t> push;
	};

	// Record a sequence of dispatches into one command buffer, inserting a
	// compute-to-compute memory barrier between each so a later step sees the
	// storage-buffer writes of earlier steps. Used for the diagonal wavefront.
	void run_wavefront(pipeline & p, const std::vector<buffer *> & bindings, const std::vector<step> & steps)
	{
		std::vector<VkDescriptorBufferInfo> infos(bindings.size());
		std::vector<VkWriteDescriptorSet> writes(bindings.size());
		for (size_t i = 0; i < bindings.size(); ++i)
		{
			infos[i] = {bindings[i]->buf, 0, VK_WHOLE_SIZE};
			writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
			writes[i].dstSet = p.dset;
			writes[i].dstBinding = (uint32_t)i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			writes[i].pBufferInfo = &infos[i];
		}
		vkUpdateDescriptorSets(dev, (uint32_t)writes.size(), writes.data(), 0, nullptr);

		VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		cbai.commandPool = pool;
		cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cbai.commandBufferCount = 1;
		VkCommandBuffer cmd;
		vkcheck(vkAllocateCommandBuffers(dev, &cbai, &cmd), "allocCmd");
		VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd, &bi);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.dset, 0, nullptr);
		VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		for (size_t s = 0; s < steps.size(); ++s)
		{
			if (s > 0)
				vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &mb, 0, nullptr, 0, nullptr);
			if (!steps[s].push.empty())
				vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)steps[s].push.size(), steps[s].push.data());
			vkCmdDispatch(cmd, steps[s].gx, steps[s].gy, steps[s].gz);
		}
		vkEndCommandBuffer(cmd);

		VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
		VkFence fence;
		vkCreateFence(dev, &fci, nullptr, &fence);
		VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
		si.commandBufferCount = 1;
		si.pCommandBuffers = &cmd;
		vkcheck(vkQueueSubmit(queue, 1, &si, fence), "queueSubmit");
		vkcheck(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "waitFence");
		vkDestroyFence(dev, fence, nullptr);
		vkFreeCommandBuffers(dev, pool, 1, &cmd);
	}

	// Bind image views (first bindings) + buffers, dispatch, block. For tests that
	// exercise the image-sampling recon path via a single dispatch.
	void run_img(pipeline & p, const std::vector<VkImageView> & imgs, const std::vector<buffer *> & bufs,
	             uint32_t gx, uint32_t gy, uint32_t gz, const void * push = nullptr, int push_bytes = 0)
	{
		bind_set_img(p, imgs, bufs);
		VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
		VkCommandBuffer cmd; vkcheck(vkAllocateCommandBuffers(dev, &cbai, &cmd), "allocCmd");
		VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd, &bi);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.dset, 0, nullptr);
		if (push && push_bytes > 0) vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
		vkCmdDispatch(cmd, gx, gy, gz);
		vkEndCommandBuffer(cmd);
		VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; vkCreateFence(dev, &fci, nullptr, &fence);
		VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
		vkcheck(vkQueueSubmit(queue, 1, &si, fence), "queueSubmit");
		vkcheck(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "waitFence");
		vkDestroyFence(dev, fence, nullptr); vkFreeCommandBuffers(dev, pool, 1, &cmd);
	}

	// Bind buffers and dispatch (gx,gy,gz workgroups). Blocks until complete.
	void run(pipeline & p, const std::vector<buffer *> & bindings, uint32_t gx, uint32_t gy, uint32_t gz,
	         const void * push = nullptr, int push_bytes = 0)
	{
		std::vector<VkDescriptorBufferInfo> infos(bindings.size());
		std::vector<VkWriteDescriptorSet> writes(bindings.size());
		for (size_t i = 0; i < bindings.size(); ++i)
		{
			infos[i] = {bindings[i]->buf, 0, VK_WHOLE_SIZE};
			writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
			writes[i].dstSet = p.dset;
			writes[i].dstBinding = (uint32_t)i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			writes[i].pBufferInfo = &infos[i];
		}
		vkUpdateDescriptorSets(dev, (uint32_t)writes.size(), writes.data(), 0, nullptr);

		VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		cbai.commandPool = pool;
		cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cbai.commandBufferCount = 1;
		VkCommandBuffer cmd;
		vkcheck(vkAllocateCommandBuffers(dev, &cbai, &cmd), "allocCmd");
		VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd, &bi);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.dset, 0, nullptr);
		if (push && push_bytes > 0)
			vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
		vkCmdDispatch(cmd, gx, gy, gz);
		vkEndCommandBuffer(cmd);

		VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
		VkFence fence;
		vkCreateFence(dev, &fci, nullptr, &fence);
		VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
		si.commandBufferCount = 1;
		si.pCommandBuffers = &cmd;
		vkcheck(vkQueueSubmit(queue, 1, &si, fence), "queueSubmit");
		vkcheck(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "waitFence");
		vkDestroyFence(dev, fence, nullptr);
		vkFreeCommandBuffers(dev, pool, 1, &cmd);
	}

	// --- Batched recording: record several dispatch groups (and buffer clears)
	// into ONE command buffer, then submit once. Each pipeline's descriptor set
	// is updated once per batch, so a pipeline must be used with a single set of
	// bindings within a batch (true for the encoder's stages). Blocking submit,
	// so the reused per-pipeline set is safe across frames. ---
	struct batch
	{
		VkCommandBuffer cmd = VK_NULL_HANDLE;
		VkQueryPool tspool = VK_NULL_HANDLE; // optional GPU timestamps
		uint32_t ts_count = 0;
		static constexpr uint32_t ts_max = 16;
		std::vector<uint64_t> ts; // filled after submit_and_wait when tspool != null
	};

private:
	void bind_set(pipeline & p, const std::vector<buffer *> & bindings)
	{
		std::vector<VkDescriptorBufferInfo> infos(bindings.size());
		std::vector<VkWriteDescriptorSet> writes(bindings.size());
		for (size_t i = 0; i < bindings.size(); ++i)
		{
			infos[i] = {bindings[i]->buf, 0, VK_WHOLE_SIZE};
			writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
			writes[i].dstSet = p.dset;
			writes[i].dstBinding = (uint32_t)i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			writes[i].pBufferInfo = &infos[i];
		}
		vkUpdateDescriptorSets(dev, (uint32_t)writes.size(), writes.data(), 0, nullptr);
	}
	// Bind image views to bindings [0, imgs.size()) as STORAGE_IMAGE (GENERAL
	// layout) and buffers to the following bindings.
	void bind_set_img(pipeline & p, const std::vector<VkImageView> & imgs, const std::vector<buffer *> & bufs)
	{
		size_t n = imgs.size() + bufs.size();
		std::vector<VkDescriptorImageInfo> iinfos(imgs.size());
		std::vector<VkDescriptorBufferInfo> binfos(bufs.size());
		std::vector<VkWriteDescriptorSet> writes(n);
		for (size_t i = 0; i < imgs.size(); ++i)
		{
			iinfos[i] = {VK_NULL_HANDLE, imgs[i], VK_IMAGE_LAYOUT_GENERAL};
			writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
			writes[i].dstSet = p.dset; writes[i].dstBinding = (uint32_t)i; writes[i].descriptorCount = 1;
			writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[i].pImageInfo = &iinfos[i];
		}
		for (size_t i = 0; i < bufs.size(); ++i)
		{
			binfos[i] = {bufs[i]->buf, 0, VK_WHOLE_SIZE};
			size_t w = imgs.size() + i;
			writes[w] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
			writes[w].dstSet = p.dset; writes[w].dstBinding = (uint32_t)w; writes[w].descriptorCount = 1;
			writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[w].pBufferInfo = &binfos[i];
		}
		vkUpdateDescriptorSets(dev, (uint32_t)writes.size(), writes.data(), 0, nullptr);
	}
	void compute_barrier(VkCommandBuffer cmd)
	{
		VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     0, 1, &mb, 0, nullptr, 0, nullptr);
	}

public:
	batch begin_batch()
	{
		batch b;
		VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		cbai.commandPool = pool;
		cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cbai.commandBufferCount = 1;
		vkcheck(vkAllocateCommandBuffers(dev, &cbai, &b.cmd), "allocCmd");
		VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(b.cmd, &bi);
		if (profile_gpu)
		{
			VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
			qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
			qpi.queryCount = batch::ts_max;
			if (vkCreateQueryPool(dev, &qpi, nullptr, &b.tspool) == VK_SUCCESS)
				vkCmdResetQueryPool(b.cmd, b.tspool, 0, batch::ts_max);
		}
		return b;
	}

	// Latch a GPU timestamp at this point in the batch (no-op unless profiling).
	// Writes at BOTTOM_OF_PIPE so it reflects completion of all prior commands.
	void record_timestamp(batch & b)
	{
		if (b.tspool && b.ts_count < batch::ts_max)
			vkCmdWriteTimestamp(b.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, b.tspool, b.ts_count++);
	}

	// Zero a buffer on the GPU, then a TRANSFER->COMPUTE barrier so later dispatches see it.
	void record_fill(batch & b, buffer & buf, size_t bytes, uint32_t value = 0)
	{
		vkCmdFillBuffer(b.cmd, buf.buf, 0, bytes, value);
		VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(b.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     0, 1, &mb, 0, nullptr, 0, nullptr);
	}

	void record_wavefront(batch & b, pipeline & p, const std::vector<buffer *> & bindings,
	                      const std::vector<step> & steps, bool leading_barrier)
	{
		bind_set(p, bindings);
		if (leading_barrier)
			compute_barrier(b.cmd);
		vkCmdBindPipeline(b.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
		vkCmdBindDescriptorSets(b.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.dset, 0, nullptr);
		for (size_t s = 0; s < steps.size(); ++s)
		{
			if (s > 0)
				compute_barrier(b.cmd);
			if (!steps[s].push.empty())
				vkCmdPushConstants(b.cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)steps[s].push.size(), steps[s].push.data());
			vkCmdDispatch(b.cmd, steps[s].gx, steps[s].gy, steps[s].gz);
		}
	}

	void record_dispatch(batch & b, pipeline & p, const std::vector<buffer *> & bindings,
	                     uint32_t gx, uint32_t gy, uint32_t gz, const void * push, int push_bytes, bool leading_barrier)
	{
		bind_set(p, bindings);
		if (leading_barrier)
			compute_barrier(b.cmd);
		vkCmdBindPipeline(b.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
		vkCmdBindDescriptorSets(b.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.dset, 0, nullptr);
		if (push && push_bytes > 0)
			vkCmdPushConstants(b.cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
		vkCmdDispatch(b.cmd, gx, gy, gz);
	}

	// Like record_dispatch, but the first bindings are STORAGE_IMAGE views (imgs)
	// and the rest are buffers. Used by the recon stage sampling the source image.
	void record_dispatch_img(batch & b, pipeline & p, const std::vector<VkImageView> & imgs, const std::vector<buffer *> & bufs,
	                         uint32_t gx, uint32_t gy, uint32_t gz, const void * push, int push_bytes, bool leading_barrier)
	{
		bind_set_img(p, imgs, bufs);
		if (leading_barrier)
			compute_barrier(b.cmd);
		vkCmdBindPipeline(b.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
		vkCmdBindDescriptorSets(b.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.dset, 0, nullptr);
		if (push && push_bytes > 0)
			vkCmdPushConstants(b.cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
		vkCmdDispatch(b.cmd, gx, gy, gz);
	}

	// End, submit, wait, free. Optionally wait on a timeline semaphore value first.
	void submit_and_wait(batch & b, VkSemaphore wait_sem = VK_NULL_HANDLE, uint64_t wait_value = 0)
	{
		vkEndCommandBuffer(b.cmd);
		VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
		VkFence fence;
		vkCreateFence(dev, &fci, nullptr, &fence);
		VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
		si.commandBufferCount = 1;
		si.pCommandBuffers = &b.cmd;
		VkTimelineSemaphoreSubmitInfo tsi{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
		VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		if (wait_sem != VK_NULL_HANDLE)
		{
			si.waitSemaphoreCount = 1;
			si.pWaitSemaphores = &wait_sem;
			si.pWaitDstStageMask = &wait_stage;
			tsi.waitSemaphoreValueCount = 1;
			tsi.pWaitSemaphoreValues = &wait_value;
			si.pNext = &tsi;
		}
		vkcheck(vkQueueSubmit(queue, 1, &si, fence), "queueSubmit");
		vkcheck(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "waitFence");
		vkDestroyFence(dev, fence, nullptr);
		vkFreeCommandBuffers(dev, pool, 1, &b.cmd);
		if (b.tspool)
		{
			b.ts.resize(b.ts_count);
			if (b.ts_count)
				vkGetQueryPoolResults(dev, b.tspool, 0, b.ts_count, b.ts_count * sizeof(uint64_t),
				                      b.ts.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			vkDestroyQueryPool(dev, b.tspool, nullptr);
			b.tspool = VK_NULL_HANDLE;
		}
	}

	// Convert a (from,to) timestamp index pair to microseconds. Returns 0 if unavailable.
	double ts_us(const batch & b, uint32_t from, uint32_t to) const
	{
		if (from >= b.ts.size() || to >= b.ts.size() || b.ts[to] < b.ts[from])
			return 0.0;
		return double(b.ts[to] - b.ts[from]) * double(timestamp_period) / 1000.0;
	}
};

} // namespace wivrn::h267::gpu
