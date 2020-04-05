#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <vector>
#include <Bitmap.h>
#include <string.h>
#include <stdexcept>
#include <cmath>

using namespace glm;

const int SAMPLES = 128;
const int WIDTH = 1024; 
const int HEIGHT = 768; 
const int WORKGROUP_SIZE = 32; 

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

#define VK_CHECK_RESULT(f) 																				\
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);																		\
    }																									\
}


class ComputeApplication {
private:
    struct Pixel {
        float r, g, b, a;
        Pixel (const vec4& color) : r(color.x), g(color.y), b(color.z), a(color.w) {}
        operator vec4()
        {
            return vec4(r, g, b, a);
        }
    };

    // Uniform data for shader to have random vector
    struct UniformBufferObject
    {
        vec2 rand_vec;
        float invSamples;
    };
    
    VkInstance instance;

    VkDebugReportCallbackEXT debugReportCallback;
    
    VkPhysicalDevice physicalDevice;
    
    VkDevice device;

    
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule computeShaderModule;

    
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    
    VkBuffer buffer;
    VkBuffer uniformBuffer;
    VkDeviceMemory uniformBufferMemory;
    VkDeviceMemory bufferMemory;
        
    uint32_t bufferSize; 

    std::vector<const char *> enabledLayers;

    
    VkQueue queue; 

    
    uint32_t queueFamilyIndex;

public:
    void run(const char* fname) {
    
        bufferSize = sizeof(Pixel) * WIDTH * HEIGHT;

        createInstance();
        findPhysicalDevice();
        createDevice();
        createBuffer();
        createUniformBuffer();
        createDescriptorSetLayout();
        createDescriptorSet();
        createComputePipeline();  
        createCommandBuffer();

        // Making sampling of path tracing. For optimization some of samples made in compute shader
        for (int i = 0; i < SAMPLES; i++)
        {

            updateUniformData(i + 1);
            runCommandBuffer();
        }

        saveRenderedImage(fname);
        cleanup();
    }

    void saveRenderedImage(const char* fname) {
        void* mappedMemory = NULL;
        
        vkMapMemory(device, bufferMemory, 0, bufferSize, 0, &mappedMemory);
        Pixel* pmappedMemory = (Pixel *)mappedMemory;

        std::vector<unsigned char> image;
        image.reserve(WIDTH * HEIGHT * 4);
        for (int i = 0; i < WIDTH*HEIGHT; i += 1) {
            vec4 color = pmappedMemory[i];

            color /= (float)SAMPLES;

            // Tone mapping
            const float exposure = 1.0;

            color = abs(vec4(1.0) - exp( -color * exposure));
            // Gamma-correction
            color = vec4(pow(vec3(color), vec3(1./ 2.2)), 1.0);


            image.push_back((unsigned char)(255.0f * (color.x)));
            image.push_back((unsigned char)(255.0f * (color.y)));
            image.push_back((unsigned char)(255.0f * (color.z)));
            image.push_back((unsigned char)(255.0f * (color.w)));
        }
        vkUnmapMemory(device, bufferMemory);
        SaveBMP(fname, (uint32_t*) image.data(), WIDTH, HEIGHT);  
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
        VkDebugReportFlagsEXT                       flags,
        VkDebugReportObjectTypeEXT                  objectType,
        uint64_t                                    object,
        size_t                                      location,
        int32_t                                     messageCode,
        const char*                                 pLayerPrefix,
        const char*                                 pMessage,
        void*                                       pUserData) {

        printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);

        return VK_FALSE;
     }

    void createInstance() {
        std::vector<const char *> enabledExtensions;

        if (enableValidationLayers) {
            
            uint32_t layerCount;
            vkEnumerateInstanceLayerProperties(&layerCount, NULL);

            std::vector<VkLayerProperties> layerProperties(layerCount);
            vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());

    
            bool foundLayer = false;
            for (VkLayerProperties prop : layerProperties) {
                
                if (strcmp("VK_LAYER_LUNARG_standard_validation", prop.layerName) == 0) {
                    foundLayer = true;
                    break;
                }

            }
            
            if (!foundLayer) {
                throw std::runtime_error("Layer VK_LAYER_LUNARG_standard_validation not supported\n");
            }
            enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation"); // Alright, we can use this layer.

            
            uint32_t extensionCount;
            
            vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, NULL);
            std::vector<VkExtensionProperties> extensionProperties(extensionCount);
            vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, extensionProperties.data());

            bool foundExtension = false;
            for (VkExtensionProperties prop : extensionProperties) {
                if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, prop.extensionName) == 0) {
                    foundExtension = true;
                    break;
                }

            }

            if (!foundExtension) {
                throw std::runtime_error("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
            }
            enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }		

        
        VkApplicationInfo applicationInfo = {};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "Path tracer";
        applicationInfo.applicationVersion = 0;
        applicationInfo.pEngineName = "awesomepathtracerengine";
        applicationInfo.engineVersion = 0;
        applicationInfo.apiVersion = VK_API_VERSION_1_0;;
        
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.flags = 0;
        createInfo.pApplicationInfo = &applicationInfo;
        
        createInfo.enabledLayerCount = enabledLayers.size();
        createInfo.ppEnabledLayerNames = enabledLayers.data();
        createInfo.enabledExtensionCount = enabledExtensions.size();
        createInfo.ppEnabledExtensionNames = enabledExtensions.data();
    
        VK_CHECK_RESULT(vkCreateInstance(
            &createInfo,
            NULL,
            &instance));


        if (enableValidationLayers) {
            VkDebugReportCallbackCreateInfoEXT createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
            createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
            createInfo.pfnCallback = &debugReportCallbackFn;

            auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
            if (vkCreateDebugReportCallbackEXT == nullptr) {
                throw std::runtime_error("Could not load vkCreateDebugReportCallbackEXT");
            }

            VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(instance, &createInfo, NULL, &debugReportCallback));
        }
    
    }

    void findPhysicalDevice() {
        
        uint32_t deviceCount;
        vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
        if (deviceCount == 0) {
            throw std::runtime_error("could not find a device with vulkan support");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (VkPhysicalDevice device : devices) {
            if (true) { 
                physicalDevice = device;
                break;
            }
        }
    }

    // Returns the index of a queue family that supports compute operations. 
    uint32_t getComputeQueueFamilyIndex() {
        uint32_t queueFamilyCount;

        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        uint32_t i = 0;
        for (; i < queueFamilies.size(); ++i) {
            VkQueueFamilyProperties props = queueFamilies[i];

            if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                break;
            }
        }

        if (i == queueFamilies.size()) {
            throw std::runtime_error("could not find a queue family that supports operations");
        }

        return i;
    }

    void createDevice() {
        
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueFamilyIndex = getComputeQueueFamilyIndex(); 
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1; 
        float queuePriorities = 1.0;   
        queueCreateInfo.pQueuePriorities = &queuePriorities;


        VkDeviceCreateInfo deviceCreateInfo = {};

        VkPhysicalDeviceFeatures deviceFeatures = {};

        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.enabledLayerCount = enabledLayers.size();  
        deviceCreateInfo.ppEnabledLayerNames = enabledLayers.data();
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo; 
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

        VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device)); 

        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    }

    
    uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memoryProperties;

        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);


        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((memoryTypeBits & (1 << i)) &&
                ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
                return i;
        }
        return -1;
    }

    void createUniformBuffer()
    {
        VkDeviceSize uniBufferSize = sizeof(UniformBufferObject);
        createBuffer(uniBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffer, uniformBufferMemory);
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CHECK_RESULT(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer));
  

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory))


        VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, bufferMemory, 0));
    }

    void createBuffer() {
        
        VkBufferCreateInfo bufferCreateInfo = {};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = bufferSize; 
        bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; 
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; 

        
        VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, NULL, &buffer)); 

        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);
        
        
        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size; 
        
        allocateInfo.memoryTypeIndex = findMemoryType(
            memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

        VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &bufferMemory)); 
        
        
        VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, bufferMemory, 0));



        void* mappedMemory = NULL;

        vkMapMemory(device, bufferMemory, 0, bufferSize, 0, &mappedMemory);
        Pixel* pmappedMemory = (Pixel *)mappedMemory;

        for(int i = 0; i < WIDTH * HEIGHT; i++)
        {
            pmappedMemory[i] = vec4(0.0);
        }

        vkUnmapMemory(device, bufferMemory);
    }

    void updateUniformData (int sampleNum)
    {
        UniformBufferObject ubo = {};
        float r1 = ((float)rand() / (RAND_MAX));
        float r2 = ((float)rand() / (RAND_MAX));

        ubo.rand_vec = vec2(r1, r2);
        ubo.invSamples = 1.0 / (float) sampleNum;

        void* data;
        vkMapMemory(device, uniformBufferMemory, 0, sizeof(ubo), 0, &data);
            memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBufferMemory);
    }

    void createDescriptorSetLayout() {

        VkDescriptorSetLayoutBinding descriptorSetLayoutBinding[2] = {};
        descriptorSetLayoutBinding[0].binding = 0; 
        descriptorSetLayoutBinding[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBinding[0].descriptorCount = 1;
        descriptorSetLayoutBinding[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBinding[1].binding = 1; 
        descriptorSetLayoutBinding[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorSetLayoutBinding[1].descriptorCount = 1;
        descriptorSetLayoutBinding[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount = 2; 
        descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBinding; 


        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));
    }

    void createDescriptorSet() {

        VkDescriptorPoolSize descriptorPoolSize[2] = {};
        descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSize[0].descriptorCount = 1;

        descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorPoolSize[1].descriptorCount = 1;

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets = 1; 
        descriptorPoolCreateInfo.poolSizeCount = 2;
        descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize[0];



        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &descriptorPool));

        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
        descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; 
        descriptorSetAllocateInfo.descriptorPool = descriptorPool; // pool to allocate from.
        descriptorSetAllocateInfo.descriptorSetCount = 1; // allocate a single descriptor set.
        descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;


        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));


        VkDescriptorBufferInfo descriptorBufferInfo[2] = {};
        descriptorBufferInfo[0].buffer = buffer;
        descriptorBufferInfo[0].offset = 0;
        descriptorBufferInfo[0].range = bufferSize;

        descriptorBufferInfo[1].buffer = uniformBuffer;
        descriptorBufferInfo[1].offset = 0;
        descriptorBufferInfo[1].range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet writeDescriptorSet[2] = {};
        writeDescriptorSet[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet[0].dstSet = descriptorSet; 
        writeDescriptorSet[0].dstBinding = 0; 
        writeDescriptorSet[0].descriptorCount = 1; 
        writeDescriptorSet[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; 
        writeDescriptorSet[0].pBufferInfo = &descriptorBufferInfo[0];
        
        writeDescriptorSet[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet[1].dstSet = descriptorSet; 
        writeDescriptorSet[1].dstBinding = 1; 
        writeDescriptorSet[1].descriptorCount = 1; 
        writeDescriptorSet[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; 
        writeDescriptorSet[1].pBufferInfo = &descriptorBufferInfo[1];

        vkUpdateDescriptorSets(device, 2, &writeDescriptorSet[0], 0, NULL);
    }

    uint32_t* readFile(uint32_t& length, const char* filename) {

        FILE* fp = fopen(filename, "rb");
        if (fp == NULL) {
            printf("Could not find or open file: %s\n", filename);
        }

        fseek(fp, 0, SEEK_END);
        long filesize = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        long filesizepadded = long(ceil(filesize / 4.0)) * 4;

        char *str = new char[filesizepadded];
        fread(str, filesize, sizeof(char), fp);
        fclose(fp);

        for (int i = filesize; i < filesizepadded; i++) {
            str[i] = 0;
        }

        length = filesizepadded;
        return (uint32_t *)str;
    }

    void createComputePipeline() {
        uint32_t filelength;

        uint32_t* code = readFile(filelength, "shaders/comp.spv");
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.pCode = code;
        createInfo.codeSize = filelength;
        
        VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &computeShaderModule));
        delete[] code;
        VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
        shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageCreateInfo.module = computeShaderModule;
        shaderStageCreateInfo.pName = "main";

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout; 
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout));
        


        VkComputePipelineCreateInfo pipelineCreateInfo = {};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stage = shaderStageCreateInfo;
        pipelineCreateInfo.layout = pipelineLayout;


        VK_CHECK_RESULT(vkCreateComputePipelines(
            device, VK_NULL_HANDLE,
            1, &pipelineCreateInfo,
            NULL, &pipeline));
    }

    void createCommandBuffer() {
        VkCommandPoolCreateInfo commandPoolCreateInfo = {};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags = 0;
        
        commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
        VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool));


        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = commandPool; 

        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1; 
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer)); 

       
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo)); 


        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
        
        vkCmdDispatch(commandBuffer, (uint32_t)ceil(WIDTH / float(WORKGROUP_SIZE)), (uint32_t)ceil(HEIGHT / float(WORKGROUP_SIZE)), 1);

        VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
    }

    void runCommandBuffer() {

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1; 
        submitInfo.pCommandBuffers = &commandBuffer; 

        /*
          We create a fence.
        */
        VkFence fence;
        VkFenceCreateInfo fenceCreateInfo = {};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = 0;
        VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, NULL, &fence));

        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
        
        VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000));
        vkDestroyFence(device, fence, NULL);
    }

    void cleanup() {

        if (enableValidationLayers) {
            // destroy callback.
            auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
            if (func == nullptr) {
                throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
            }
            func(instance, debugReportCallback, NULL);
        }

        vkFreeMemory(device, uniformBufferMemory, NULL);
        vkFreeMemory(device, bufferMemory, NULL);
        vkDestroyBuffer(device, uniformBuffer, NULL);	
        vkDestroyBuffer(device, buffer, NULL);	
        vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);	
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);		
    }
};
