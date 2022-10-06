#include <iostream>
#include <set>
#include <fstream>

#include <SDL.h>
#include <SDL_vulkan.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "glad/vulkan.h"

VkInstance instance;
VkPhysicalDevice physical_device;
VkDevice device;

VkDebugUtilsMessengerEXT debug_messenger;

uint32_t graphics_family, present_family;
VkQueue graphics_queue, present_queue;

VkSurfaceKHR surface;

VkSwapchainKHR swap_chain;
std::vector<VkImage> swap_chain_images;
VkFormat swap_chain_format;
VkExtent2D swap_chain_extent;

std::vector<VkImageView> image_views;

std::vector<VkShaderModule> shader_stages;

VkRenderPass render_pass;
VkPipelineLayout pipeline_layout;

VkPipeline graphics_pipeline;

std::vector<VkFramebuffer> frame_buffers;

VkCommandPool command_pool;
VkCommandBuffer command_buffer;

VkSemaphore image_available, render_finished;
VkFence image_in_flight;

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                     VkDebugUtilsMessageTypeFlagsEXT type,
                                                     const VkDebugUtilsMessengerCallbackDataEXT *data,
                                                     void *user_data)
{
    spdlog::error("Validation Layer: {}", data->pMessage);
    return VK_FALSE;
}

struct SwapChainDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
};

void create_sync_objs()
{
    VkSemaphoreCreateInfo semaphore_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT};

    vkCreateSemaphore(device, &semaphore_info, NULL, &image_available);
    vkCreateSemaphore(device, &semaphore_info, NULL, &render_finished);
    vkCreateFence(device, &fence_info, NULL, &image_in_flight);
}

static std::vector<char> readShader(const char *path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    size_t size = file.tellg();
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), size);
    file.close();

    return buffer;
}

VkShaderModule create_shader(const std::vector<char> &code)
{
    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t *>(code.data())};

    VkShaderModule module;
    if (vkCreateShaderModule(device, &create_info, NULL, &module) != VK_SUCCESS)
        spdlog::error("Failed to create shader module");

    return module;
}

void create_renderpass()
{
    VkAttachmentDescription color_attachment = {
        .format = swap_chain_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR};

    VkAttachmentReference color_attachment_ref = {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref};

    VkSubpassDependency dependency = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT};

    VkRenderPassCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency};

    if (vkCreateRenderPass(device, &create_info, NULL, &render_pass) != VK_SUCCESS)
        spdlog::error("Failed to create render pass!");
}

void create_graphics_pipeline()
{
    auto vertex_code = readShader("assets/vert.spv");
    auto fragment_code = readShader("assets/frag.spv");

    VkShaderModule vertex = create_shader(vertex_code);
    VkShaderModule fragment = create_shader(fragment_code);

    VkPipelineShaderStageCreateInfo vert_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertex,
        .pName = "main"};

    VkPipelineShaderStageCreateInfo frag_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fragment,
        .pName = "main"};

    VkPipelineShaderStageCreateInfo stages[] = {vert_info, frag_info};

    VkPipelineVertexInputStateCreateInfo vert_input_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 0,
        .vertexAttributeDescriptionCount = 0};

    VkPipelineInputAssemblyStateCreateInfo input_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE};

    VkViewport viewport = {
        .x = 0.f,
        .y = 0.f,
        .width = (float)swap_chain_extent.width,
        .height = (float)swap_chain_extent.height,
        .minDepth = 0.f,
        .maxDepth = 1.f};

    VkRect2D scissor = {
        .offset = {0, 0},
        .extent = swap_chain_extent};

    std::vector<VkDynamicState> dynamic_states = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamic_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data()};

    VkPipelineViewportStateCreateInfo viewport_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor};

    VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.f};

    VkPipelineMultisampleStateCreateInfo multisample = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE};

    VkPipelineColorBlendAttachmentState color_blend = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineColorBlendStateCreateInfo color_blending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &color_blend};

    VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};

    vkCreatePipelineLayout(device, &pipeline_layout_info, NULL, &pipeline_layout);

    create_renderpass();

    VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vert_input_info,
        .pInputAssemblyState = &input_info,
        .pViewportState = &viewport_info,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisample,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_state,
        .layout = pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0};

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &graphics_pipeline) != VK_SUCCESS)
        spdlog::error("Failed to create graphics pipeline");
}

SwapChainDetails query_swap_chain_details()
{
    SwapChainDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &details.capabilities);

    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, NULL);
    details.formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, details.formats.data());

    uint32_t present_mode_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, NULL);
    details.present_modes.resize(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, details.present_modes.data());

    return details;
}

VkSurfaceFormatKHR choose_format(const std::vector<VkSurfaceFormatKHR> &available_formats)
{
    for (auto format : available_formats)
    {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return format;
    }

    return available_formats[0];
}

VkPresentModeKHR choose_present_mode(const std::vector<VkPresentModeKHR> &available_modes)
{
    for (auto mode : available_modes)
    {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
            return mode;
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D choose_extent(const VkSurfaceCapabilitiesKHR &capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        return capabilities.currentExtent;
    else
    {
        uint32_t width = 1280, height = 720;

        VkExtent2D actual_extent = {
            width, height};

        actual_extent.width = std::clamp(actual_extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actual_extent.height = std::clamp(actual_extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actual_extent;
    }
}

void create_framebuffers()
{
    frame_buffers.resize(image_views.size());

    for (int i = 0; i < image_views.size(); i++)
    {
        VkImageView attachments[] = {
            image_views[i]};

        VkFramebufferCreateInfo fb_info = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = attachments,
            .width = swap_chain_extent.width,
            .height = swap_chain_extent.height,
            .layers = 1};

        if (vkCreateFramebuffer(device, &fb_info, NULL, &frame_buffers[i]) != VK_SUCCESS)
            spdlog::error("Failed to create framebuffer #{}", i);
    }
}

void create_commandpool()
{
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = graphics_family};

    if (vkCreateCommandPool(device, &pool_info, NULL, &command_pool) != VK_SUCCESS)
        spdlog::error("Failed to create command pool");
}

void create_commandbuffer()
{
    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};

    vkAllocateCommandBuffers(device, &alloc_info, &command_buffer);
}

void record_command_buffer(uint32_t image)
{
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };

    vkBeginCommandBuffer(command_buffer, &begin_info);

    VkClearValue clear_color = {{{0.f, 0.f, 0.f, 1.f}}};
    VkRenderPassBeginInfo render_pass_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = render_pass,
        .framebuffer = frame_buffers[image],
        .renderArea = {
            .offset = {0, 0},
            .extent = swap_chain_extent},
        .clearValueCount = 1,
        .pClearValues = &clear_color};

    vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

    VkViewport viewport = {
        .x = 0.f,
        .y = 0.f,
        .width = static_cast<float>(swap_chain_extent.width),
        .height = static_cast<float>(swap_chain_extent.height),
        .minDepth = 0.f,
        .maxDepth = 1.f};

    vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    VkRect2D scissor = {
        .offset = {0, 0},
        .extent = swap_chain_extent};

    vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    vkCmdDraw(command_buffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(command_buffer);

    vkEndCommandBuffer(command_buffer);
}

void init_vulkan(SDL_Window *window)
{
    gladLoaderLoadVulkan(NULL, NULL, NULL);

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "YAMCC",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "YAMCC",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0};

    uint32_t extension_count = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &extension_count, NULL);
    std::vector<const char *> extensions(extension_count);
    SDL_Vulkan_GetInstanceExtensions(window, &extension_count, extensions.data());
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    for (auto ext : extensions)
        spdlog::debug("Vulkan Extension: {}", ext);

    std::vector<const char *> validation_layers = {
        "VK_LAYER_KHRONOS_validation"};

    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = static_cast<uint32_t>(validation_layers.size()),
        .ppEnabledLayerNames = validation_layers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data()};

    VkResult result = vkCreateInstance(&create_info, NULL, &instance);
    if (result != VK_SUCCESS)
        spdlog::error("Failed to create Vulkan instance!");

    gladLoaderLoadVulkan(instance, NULL, NULL);

    VkDebugUtilsMessengerCreateInfoEXT debug_info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debug_callback,
        .pUserData = NULL};

    vkCreateDebugUtilsMessengerEXT(instance, &debug_info, NULL, &debug_messenger);

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, NULL);
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

    // TODO(skettios): find most suitable device
    physical_device = devices[0];

    gladLoaderLoadVulkan(instance, physical_device, NULL);

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, NULL);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());

    if (!SDL_Vulkan_CreateSurface(window, instance, &surface))
        spdlog::error("Failed to create vulkan surface!");

    for (int i = 0; i < queue_families.size(); i++)
    {
        if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            graphics_family = i;

        VkBool32 present_support = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, surface, &present_support);
        if (present_support)
            present_family = i;
    }

    std::set<uint32_t> unique_queue_families = {graphics_family, present_family};
    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;

    float queue_priority = 1.f;
    for (uint32_t queue_family : unique_queue_families)
    {
        VkDeviceQueueCreateInfo queue_create_info = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queue_family,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority};

        queue_create_infos.push_back(queue_create_info);
    }

    VkPhysicalDeviceFeatures device_features = {

    };

    const std::vector<const char *> device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkDeviceCreateInfo device_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size()),
        .pQueueCreateInfos = queue_create_infos.data(),
        .enabledLayerCount = 0,
        .enabledExtensionCount = static_cast<uint32_t>(device_extensions.size()),
        .ppEnabledExtensionNames = device_extensions.data(),
        .pEnabledFeatures = &device_features};

    result = vkCreateDevice(physical_device, &device_create_info, NULL, &device);
    if (result != VK_SUCCESS)
        spdlog::error("Failed to create logical Vulkan device!");

    gladLoaderLoadVulkan(instance, physical_device, device);

    vkGetDeviceQueue(device, graphics_family, 0, &graphics_queue);
    vkGetDeviceQueue(device, present_family, 0, &present_queue);

    SwapChainDetails swap_chain_details = query_swap_chain_details();

    VkSurfaceFormatKHR surface_format = choose_format(swap_chain_details.formats);
    VkPresentModeKHR present_mode = choose_present_mode(swap_chain_details.present_modes);
    VkExtent2D extent = choose_extent(swap_chain_details.capabilities);

    uint32_t image_count = swap_chain_details.capabilities.minImageCount + 1;
    if (swap_chain_details.capabilities.maxImageCount > 0 && image_count > swap_chain_details.capabilities.maxImageCount)
        image_count = swap_chain_details.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR swap_chain_create_info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};

    uint32_t queue_family_indices[] = {graphics_family, present_family};
    if (graphics_family != present_family)
    {
        swap_chain_create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swap_chain_create_info.queueFamilyIndexCount = 2;
        swap_chain_create_info.pQueueFamilyIndices = queue_family_indices;
    }
    else
    {
        swap_chain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    swap_chain_create_info.preTransform = swap_chain_details.capabilities.currentTransform;
    swap_chain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swap_chain_create_info.presentMode = present_mode;
    swap_chain_create_info.clipped = VK_TRUE;
    swap_chain_create_info.oldSwapchain = VK_NULL_HANDLE;

    result = vkCreateSwapchainKHR(device, &swap_chain_create_info, NULL, &swap_chain);
    if (result != VK_SUCCESS)
        spdlog::error("Failed to create swapchain!");

    vkGetSwapchainImagesKHR(device, swap_chain, &image_count, NULL);
    swap_chain_images.resize(image_count);
    vkGetSwapchainImagesKHR(device, swap_chain, &image_count, swap_chain_images.data());

    swap_chain_format = surface_format.format;
    swap_chain_extent = extent;

    image_views.resize(swap_chain_images.size());

    for (size_t i = 0; i < swap_chain_images.size(); i++)
    {
        VkImageViewCreateInfo iv_create_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = swap_chain_images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = swap_chain_format,
            .components = {
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY},
            .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};

        result = vkCreateImageView(device, &iv_create_info, NULL, &image_views[i]);
        if (result != VK_SUCCESS)
            spdlog::error("Failed to create Image View #{}", i);
    }

    create_graphics_pipeline();
    create_framebuffers();
    create_commandpool();
    create_commandbuffer();
    create_sync_objs();
}

void draw_frame()
{
    vkWaitForFences(device, 1, &image_in_flight, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &image_in_flight);

    uint32_t image_index;
    vkAcquireNextImageKHR(device, swap_chain, UINT64_MAX, image_available, VK_NULL_HANDLE, &image_index);

    vkResetCommandBuffer(command_buffer, 0);
    record_command_buffer(image_index);

    VkSemaphore wait[] = {image_available};
    VkSemaphore signal[] = {render_finished};
    VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = wait,
        .pWaitDstStageMask = wait_stages,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = signal};

    vkQueueSubmit(graphics_queue, 1, &submit_info, image_in_flight);

    VkSwapchainKHR swapchains[] = {swap_chain};
    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = signal,
        .swapchainCount = 1,
        .pSwapchains = swapchains,
        .pImageIndices = &image_index,
        .pResults = NULL};

    vkQueuePresentKHR(present_queue, &present_info);
}

int main(int argc, char **argv)
{
    try
    {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::err);

        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("yamcc.log", true);
        file_sink->set_level(spdlog::level::err);

        auto default_logger = std::make_shared<spdlog::logger>(spdlog::logger("YMCC", {console_sink, file_sink}));
        spdlog::set_default_logger(default_logger);
        spdlog::set_level(spdlog::level::err);
    }
    catch (const std::exception &e)
    {
        std::cout << "Log init failed: " << e.what() << std::endl;
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        return -1;

    SDL_Window *window = SDL_CreateWindow("YAMCC",
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          1280,
                                          720,
                                          SDL_WINDOW_VULKAN);
    if (!window)
        return -1;

    init_vulkan(window);

    bool running = true;
    while (running)
    {
        SDL_Event window_event;
        while (SDL_PollEvent(&window_event))
        {
            switch (window_event.type)
            {
            case SDL_QUIT:
                running = false;
                break;
            default:
                break;
            }
        }

        draw_frame();
    }

    vkDeviceWaitIdle(device);

    for (auto iv : image_views)
        vkDestroyImageView(device, iv, NULL);

    vkDestroySwapchainKHR(device, swap_chain, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroySurfaceKHR(instance, surface, NULL);
    vkDestroyInstance(instance, NULL);

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
