#pragma once

#include "ggml.h"
#include "llama.h"

#include <functional>
#include <string>
#include <vector>

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// a second model that shares the devices of the main model, e.g. a draft model
//   - its context follows the context of the main model, so its memory is measured again whenever that context changes
//   - shares_model tells the fit that the weights are already counted in the main model, as for an MTP context
struct common_fit_extra_model {
    const char * path_model;
    llama_model_params * mparams;
    llama_context_params * cparams;
    bool shares_model;
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
common_params_fit_status common_fit_params(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                             size_t * margins,               // margins of memory to leave per device in bytes
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
      const common_fit_extra_model * extra,                  // model to fit alongside the main one, nullptr if there is none
                     ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

// print estimated memory to stdout
void common_fit_print(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

void common_memory_breakdown_print(const llama_context * ctx);

// tensor split: piano trovato dalla ricerca su (n_ctx, n_ubatch)
struct common_fit_tensor_plan {
    uint32_t n_ctx    = 0;
    uint32_t n_ubatch = 0;
    bool     fits     = false;
    std::vector<std::string> notes; // passi della ricerca, leggibili
};

// Ricerca pura (senza modello caricato) per il tensor split: i pesi non si spostano e lo split e' fisso, le leve sono
// l'ubatch (dimezzato per primo, fino a n_ubatch_min) e poi il contesto (interpolato per scheda fra n_ctx_min e n_ctx,
// allineato a n_ctx_align, mai sotto n_ctx_min; non toccato se n_ctx_locked). used_per_dev(n_ctx, n_ubatch) restituisce
// i byte previsti per ogni scheda; entra quando used + margin <= free su tutte.
common_fit_tensor_plan common_fit_tensor_search(
    const std::function<std::vector<int64_t>(uint32_t n_ctx, uint32_t n_ubatch)> & used_per_dev,
    const std::vector<int64_t> & free_per_dev,
    const std::vector<int64_t> & margins_per_dev,
    uint32_t n_ctx, uint32_t n_ubatch,
    uint32_t n_ctx_min, uint32_t n_ctx_align,
    uint32_t n_ubatch_min, bool n_ctx_locked);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);
