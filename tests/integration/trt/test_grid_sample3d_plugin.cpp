#include <dlfcn.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <NvInfer.h>

#include "grid_sample_3d_plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace {

class TestLogger final : public ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << message << std::endl;
        }
    }
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::filesystem::path plugin_library_path(const char* argv0) {
    (void)argv0;
    return std::filesystem::read_symlink("/proc/self/exe").parent_path().parent_path().parent_path().parent_path() / "libgrid_sample_3d_plugin.so";
}

void load_plugin_library(const char* argv0) {
    auto plugin_path = plugin_library_path(argv0);
    void* handle = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    const char* error_message = dlerror();
    require(handle != nullptr, std::string("failed to dlopen plugin library: ") + (error_message ? error_message : "unknown error"));
}

IPluginCreator* get_creator() {
    auto* registry = getPluginRegistry();
    require(registry != nullptr, "plugin registry unavailable");
    auto* creator = registry->getCreator("GridSample3D", "1", "");
    require(creator != nullptr, "GridSample3D creator not registered");
    auto* typed_creator = dynamic_cast<IPluginCreator*>(creator);
    require(typed_creator != nullptr, "GridSample3D creator type cast failed");
    return typed_creator;
}

void destroy_plugin(IPluginV2* value) {
    value->destroy();
}

std::unique_ptr<IPluginV2, void (*)(IPluginV2*)> make_plugin(IPluginCreator* creator);

void validate_output_dimensions(IPluginCreator* creator, ILogger& logger) {
    auto builder = std::unique_ptr<IBuilder, void (*)(IBuilder*)>(createInferBuilder(logger), [](IBuilder* value) {
        if (value != nullptr) {
            delete value;
        }
    });
    require(builder != nullptr, "failed to create TensorRT builder");

    auto network = std::unique_ptr<INetworkDefinition, void (*)(INetworkDefinition*)>(builder->createNetworkV2(0), [](INetworkDefinition* value) {
        if (value != nullptr) {
            delete value;
        }
    });
    require(network != nullptr, "failed to create TensorRT network");

    auto* input = network->addInput("input", DataType::kFLOAT, Dims{5, {1, 2, 2, 2, 2}});
    auto* grid = network->addInput("grid", DataType::kFLOAT, Dims{5, {1, 1, 1, 2, 3}});
    require(input != nullptr && grid != nullptr, "failed to add network inputs");

    auto plugin = make_plugin(creator);
    ITensor* inputs[] = {input, grid};
    auto* layer = network->addPluginV2(inputs, 2, *plugin);
    require(layer != nullptr, "failed to add plugin layer to network");

    auto dims = layer->getOutput(0)->getDimensions();
    require(dims.nbDims == 5, "unexpected output rank");
    require(dims.d[0] == 1 && dims.d[1] == 2 && dims.d[2] == 1 && dims.d[3] == 1 && dims.d[4] == 2, "unexpected output dimensions");
}

std::unique_ptr<IPluginV2, void (*)(IPluginV2*)> make_plugin(IPluginCreator* creator) {
    int interpolation_mode = 0;
    int padding_mode = 1;
    int align_corners = 1;

    PluginField interpolation_field{"interpolation_mode", &interpolation_mode, PluginFieldType::kINT32, 1};
    PluginField padding_field{"padding_mode", &padding_mode, PluginFieldType::kINT32, 1};
    PluginField align_field{"align_corners", &align_corners, PluginFieldType::kINT32, 1};
    PluginField fields[] = {interpolation_field, padding_field, align_field};
    PluginFieldCollection collection{3, fields};

    auto* plugin = creator->createPlugin("grid_sample_3d", &collection);
    require(plugin != nullptr, "failed to create plugin");
    return {plugin, destroy_plugin};
}

void validate_precision_contract(GridSample3DPlugin* plugin) {
    PluginTensorDesc in_out[3]{};
    for (DataType data_type : {DataType::kFLOAT, DataType::kHALF}) {
        for (auto& desc : in_out) {
            desc.format = TensorFormat::kLINEAR;
            desc.type = data_type;
        }

        require(plugin->supportsFormatCombination(0, in_out, 2, 1), "valid precision rejected at input");
        require(plugin->supportsFormatCombination(1, in_out, 2, 1), "valid precision rejected at grid");
        require(plugin->supportsFormatCombination(2, in_out, 2, 1), "valid precision rejected at output");

        DataType input_types[] = {data_type, data_type};
        require(plugin->getOutputDataType(0, input_types, 2) == data_type,
                "output precision does not follow input precision");
    }

    in_out[0].type = DataType::kFLOAT;
    in_out[1].type = DataType::kHALF;
    in_out[2].type = DataType::kFLOAT;
    require(!plugin->supportsFormatCombination(1, in_out, 2, 1),
            "mixed input and grid precision was accepted");
}

} // namespace

int main(int argc, char** argv) {
    try {
        TestLogger logger;
        require(argc > 0, "argv unavailable");
        load_plugin_library(argv[0]);
        initLibNvInferPlugins(&logger, "");

        auto* creator = get_creator();
        auto plugin = make_plugin(creator);

        require(std::string(plugin->getPluginType()) == "GridSample3D", "unexpected plugin type");
        require(std::string(plugin->getPluginVersion()) == "1", "unexpected plugin version");
        require(plugin->getNbOutputs() == 1, "unexpected output count");
        require(plugin->getSerializationSize() > 0, "serialization size should be positive");

        auto* typed_plugin = dynamic_cast<GridSample3DPlugin*>(plugin.get());
        require(typed_plugin != nullptr, "plugin type cast failed");

        validate_precision_contract(typed_plugin);

        validate_output_dimensions(creator, logger);

        std::vector<char> serialized(plugin->getSerializationSize());
        plugin->serialize(serialized.data());
        auto* deserialized = creator->deserializePlugin("grid_sample_3d", serialized.data(), serialized.size());
        require(deserialized != nullptr, "deserializePlugin failed");
        require(deserialized->getSerializationSize() == plugin->getSerializationSize(), "serialization size changed after deserialize");
        deserialized->destroy();

        auto* clone = plugin->clone();
        require(clone != nullptr, "clone failed");
        require(std::string(clone->getPluginType()) == "GridSample3D", "clone plugin type mismatch");
        clone->destroy();
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }

    std::cout << "test_grid_sample3d_plugin passed" << std::endl;
    return 0;
}