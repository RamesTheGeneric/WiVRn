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
	bool owns_device = false;          // true when init() created the device/instance

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

		VkPhysicalDeviceMemoryProperties mp;
		vkGetPhysicalDeviceMemoryProperties(phys, &mp);
		const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		const VkMemoryPropertyFlags want_cached = want | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
		for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
		{
			auto f = mp.memoryTypes[i].propertyFlags;
			if (host_mem_type == ~0u && (f & want) == want)
				host_mem_type = i;
			// Prefer a coherent+cached type: fast CPU reads with no manual
			// invalidation. On this APU the plain coherent type is write-combined,
			// which reads at ~100 MB/s and dominated the encode time.
			if (host_read_mem_type == ~0u && (f & want_cached) == want_cached)
				host_read_mem_type = i;
		}
		if (host_mem_type == ~0u)
			throw std::runtime_error("no host-visible coherent memory");
		if (host_read_mem_type == ~0u)
			host_read_mem_type = host_mem_type; // no cached type; fall back
	}

public:

	// cached=true picks HOST_CACHED memory (fast CPU reads) for buffers the CPU
	// reads back; leave false for write-mostly upload buffers.
	buffer make_buffer(size_t bytes, bool cached = false)
	{
		buffer b;
		b.size = bytes;
		VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bci.size = bytes;
		bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		vkcheck(vkCreateBuffer(dev, &bci, nullptr, &b.buf), "vkCreateBuffer");
		VkMemoryRequirements req;
		vkGetBufferMemoryRequirements(dev, b.buf, &req);
		VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		mai.allocationSize = req.size;
		mai.memoryTypeIndex = cached ? host_read_mem_type : host_mem_type;
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

	pipeline make_pipeline(const char * spv_path, int nbindings, int push_bytes = 0)
	{
		auto code = read_spv(spv_path);
		return make_pipeline_from_code(code.data(), code.size(), nbindings, push_bytes);
	}

	// Build a compute pipeline from SPIR-V words already in memory (the shape used
	// inside WiVRn, where shaders are embedded as a std::map<string,vector<u32>>).
	pipeline make_pipeline_from_code(const uint32_t * code, size_t words, int nbindings, int push_bytes = 0)
	{
		pipeline p;
		p.nbindings = nbindings;
		p.push_bytes = push_bytes;
		VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
		smci.codeSize = words * 4;
		smci.pCode = code;
		vkcheck(vkCreateShaderModule(dev, &smci, nullptr, &p.shader), "vkCreateShaderModule");

		std::vector<VkDescriptorSetLayoutBinding> binds(nbindings);
		for (int i = 0; i < nbindings; ++i)
			binds[i] = {(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
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

		VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)nbindings};
		VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		dpci.maxSets = 1;
		dpci.poolSizeCount = 1;
		dpci.pPoolSizes = &ps;
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
};

} // namespace wivrn::h267::gpu
