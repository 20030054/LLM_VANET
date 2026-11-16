//
// Blocking TCP client for the decision server (engine/kvrsu/server.py).
//
#pragma once

#include <string>

namespace kvrsu {

class EngineClient {
public:
    ~EngineClient();
    // Tries to connect until `timeoutSec` has passed (the server may still be loading the model).
    void connectTo(const std::string& host, int port, double timeoutSec);
    // Sends `request` and returns everything up to and including the line "END".
    std::string exchange(const std::string& request);
    void close();
    bool connected() const { return fd >= 0; }

private:
    int fd = -1;
    std::string pending;
};

} // namespace kvrsu
