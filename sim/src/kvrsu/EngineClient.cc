#include "kvrsu/EngineClient.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace kvrsu {

EngineClient::~EngineClient()
{
    close();
}

void EngineClient::close()
{
    if (fd >= 0) {
        const char bye[] = "BYE\n";
        (void) !::send(fd, bye, sizeof(bye) - 1, MSG_NOSIGNAL);
        ::close(fd);
        fd = -1;
    }
}

void EngineClient::connectTo(const std::string& host, int port, double timeoutSec)
{
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res)
        throw std::runtime_error("EngineClient: cannot resolve " + host);

    auto start = std::chrono::steady_clock::now();
    while (true) {
        fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd >= 0 && ::connect(fd, res->ai_addr, res->ai_addrlen) == 0) break;
        if (fd >= 0) ::close(fd);
        fd = -1;
        double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (waited > timeoutSec) {
            freeaddrinfo(res);
            throw std::runtime_error("EngineClient: no decision server at " + host + ":" + std::to_string(port));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    freeaddrinfo(res);
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

std::string EngineClient::exchange(const std::string& request)
{
    size_t sent = 0;
    while (sent < request.size()) {
        ssize_t n = ::send(fd, request.data() + sent, request.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) throw std::runtime_error("EngineClient: send failed (decision server gone?)");
        sent += (size_t) n;
    }
    char buf[65536];
    while (true) {
        size_t pos = pending.find("\nEND\n");
        if (pos != std::string::npos) {
            std::string reply = pending.substr(0, pos + 5);
            pending.erase(0, pos + 5);
            return reply;
        }
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) throw std::runtime_error("EngineClient: connection closed by decision server");
        pending.append(buf, (size_t) n);
    }
}

} // namespace kvrsu
