// Test della ricerca (n_ctx, n_ubatch) di common_fit_tensor_search per il tensor split.
// Modello sintetico: due schede (4090 24 GiB, V100 32 GiB), pesi 15 GiB per scheda, cache KV lineare nel contesto,
// buffer di calcolo lineare nell'ubatch. Nessuna GPU necessaria.

#include "fit.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

constexpr int64_t GiB = 1024ll*1024*1024;
constexpr int64_t MiB = 1024ll*1024;

struct scenario {
    std::vector<int64_t> model;       // per scheda
    std::vector<int64_t> kv_per_tok;  // per scheda
    std::vector<int64_t> comp_per_ub; // per scheda
    std::vector<int64_t> comp_base;
    std::function<std::vector<int64_t>(uint32_t, uint32_t)> used() const {
        return [this](uint32_t n_ctx, uint32_t n_ub) {
            std::vector<int64_t> ret(model.size());
            for (size_t i = 0; i < model.size(); i++) {
                ret[i] = model[i] + kv_per_tok[i] * (int64_t) n_ctx + comp_base[i] + comp_per_ub[i] * (int64_t) n_ub;
            }
            return ret;
        };
    }
};

static int n_fail = 0;
#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FALLITO riga %d: %s\n", __LINE__, #cond); n_fail++; } } while (0)

static bool fits(const std::vector<int64_t> & used, const std::vector<int64_t> & free, const std::vector<int64_t> & margins) {
    for (size_t i = 0; i < used.size(); i++) {
        if (used[i] + margins[i] > free[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    const std::vector<int64_t> free    = {24*GiB - 600*MiB, 32*GiB - 600*MiB};
    const std::vector<int64_t> margins = {1*GiB, 1*GiB};
    const uint32_t align = 256;

    // 1) entra cosi' com'e': nessuna modifica
    {
        scenario s = {{15*GiB, 15*GiB}, {32*1024, 32*1024}, {512*1024, 512*1024}, {200*MiB, 200*MiB}};
        auto plan = common_fit_tensor_search(s.used(), free, margins, 131072, 512, 4096, align, 512, false);
        CHECK(plan.fits);
        CHECK(plan.n_ctx == 131072);
        CHECK(plan.n_ubatch == 512);
    }

    // 2) il buffer di calcolo a ub 2048 non entra sulla 4090, a 1024 si': si dimezza l'ubatch, il contesto resta
    {
        scenario s = {{15*GiB, 15*GiB}, {32*1024, 32*1024}, {2*MiB, 2*MiB}, {200*MiB, 200*MiB}};
        // a 131072 la cache pesa 4 GiB: 15 + 4 + 0,2 + 2*2048/1024 = 23,2 + margine 1 > 23,4  -> non entra a 2048; a 1024: 21,2 ok
        auto plan = common_fit_tensor_search(s.used(), free, margins, 131072, 2048, 4096, align, 512, false);
        CHECK(plan.fits);
        CHECK(plan.n_ctx == 131072);
        CHECK(plan.n_ubatch == 1024);
    }

    // 3) ub gia' al minimo, 262144 non entra sulla 4090: il contesto scende al massimo che entra, allineato a 256
    {
        scenario s = {{15*GiB, 15*GiB}, {32*1024, 32*1024}, {512*1024, 512*1024}, {200*MiB, 200*MiB}};
        auto plan = common_fit_tensor_search(s.used(), free, margins, 262144, 512, 4096, align, 512, false);
        CHECK(plan.fits);
        CHECK(plan.n_ubatch == 512);
        CHECK(plan.n_ctx % align == 0);
        CHECK(plan.n_ctx >= 4096 && plan.n_ctx < 262144);
        CHECK(fits(s.used()(plan.n_ctx, plan.n_ubatch), free, margins));
        // massimale entro il 5%: un passo di 5% in piu' non deve entrare
        CHECK(!fits(s.used()(plan.n_ctx + plan.n_ctx/20 + align, plan.n_ubatch), free, margins));
        // il conto a mano: (23,4 - 1 - 15 - 0,2 - 0,25) GiB / 32 KiB = 222.720 token
        CHECK(plan.n_ctx > 210000);
    }

    // 4) contesto bloccato dall'utente (-c 0 = tutto il contesto): niente riduzione, non entra
    {
        scenario s = {{15*GiB, 15*GiB}, {32*1024, 32*1024}, {512*1024, 512*1024}, {200*MiB, 200*MiB}};
        auto plan = common_fit_tensor_search(s.used(), free, margins, 262144, 512, 4096, align, 512, true);
        CHECK(!plan.fits);
        CHECK(plan.n_ctx == 262144);
    }

    // 5) non entra nemmeno al contesto minimo: fallisce e riporta il minimo
    {
        scenario s = {{24*GiB, 15*GiB}, {32*1024, 32*1024}, {512*1024, 512*1024}, {200*MiB, 200*MiB}};
        auto plan = common_fit_tensor_search(s.used(), free, margins, 262144, 512, 4096, align, 512, false);
        CHECK(!plan.fits);
        CHECK(plan.n_ctx == 4096);
        CHECK(plan.n_ubatch == 512);
    }

    // 6) cache mista: la V100 a f16 pesa il doppio per token ma ha piu' spazio; il limite resta la 4090
    {
        scenario s = {{15*GiB, 15*GiB}, {16*1024, 32*1024}, {512*1024, 512*1024}, {200*MiB, 200*MiB}};
        auto plan = common_fit_tensor_search(s.used(), free, margins, 262144, 512, 4096, align, 512, false);
        CHECK(plan.fits);
        CHECK(plan.n_ctx == 262144); // 15 + 4 + 0,45 + 1 = 20,45 < 23,4 sulla 4090; 15 + 8 + 0,45 + 1 = 24,45 < 31,4 sulla V100
    }

    // 7) prima si prova l'ubatch, poi il contesto: con ub 2048 e 262144 serve ridurre entrambi
    {
        scenario s = {{15*GiB, 15*GiB}, {32*1024, 32*1024}, {2*MiB, 2*MiB}, {200*MiB, 200*MiB}};
        auto plan = common_fit_tensor_search(s.used(), free, margins, 262144, 2048, 4096, align, 512, false);
        CHECK(plan.fits);
        CHECK(plan.n_ubatch == 512);
        CHECK(plan.n_ctx < 262144 && plan.n_ctx % align == 0);
        CHECK(fits(s.used()(plan.n_ctx, plan.n_ubatch), free, margins));
    }

    if (n_fail == 0) {
        printf("test-fit-tensor: tutto OK\n");
        return 0;
    }
    printf("test-fit-tensor: %d controlli falliti\n", n_fail);
    return 1;
}
