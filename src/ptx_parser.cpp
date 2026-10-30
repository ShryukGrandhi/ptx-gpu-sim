// Parser for the PTX subset executed by the simulator.
//
// Handles the shape of hand-written PTX and of `nvcc -ptx` output for simple
// kernels: .entry functions with .param lists, .reg/.shared declarations,
// labels, predicated instructions, vector operands and address expressions.
#include "ptx.h"

#include "cfg.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>

namespace ptxsim {

int type_bytes(Type t) {
    switch (t) {
        case Type::U8: case Type::S8: case Type::B8: return 1;
        case Type::U16: case Type::S16: case Type::B16: return 2;
        case Type::U32: case Type::S32: case Type::B32: case Type::F32: return 4;
        case Type::U64: case Type::S64: case Type::B64: return 8;
        case Type::Pred: return 1;
        default: return 0;
    }
}

bool type_is_signed(Type t) { return t == Type::S8 || t == Type::S16 || t == Type::S32 || t == Type::S64; }
bool type_is_float(Type t) { return t == Type::F32; }

int Kernel::estimated_regs_per_thread() const {
    int n = 0;
    for (const auto& r : regs) {
        int b = type_bytes(r.type);
        if (r.type == Type::Pred) continue;
        n += b > 4 ? 2 : 1;
    }
    return std::min(std::max(n, 1), 255);
}

const Kernel& Module::kernel(const std::string& name) const {
    for (const auto& k : kernels)
        if (k.name == name) return k;
    throw std::runtime_error("kernel not found: " + name);
}

namespace {

struct Token {
    enum Kind { Ident, Number, Punct, End } kind;
    std::string text;
    int line;
};

bool is_ident_start(char c) { return std::isalpha((unsigned char)c) || c == '_' || c == '$' || c == '%' || c == '.'; }
bool is_ident_char(char c) { return std::isalnum((unsigned char)c) || c == '_' || c == '$' || c == '.'; }

std::vector<Token> lex(const std::string& src) {
    std::vector<Token> out;
    int line = 1;
    size_t i = 0, n = src.size();
    while (i < n) {
        char c = src[i];
        if (c == '\n') { ++line; ++i; continue; }
        if (std::isspace((unsigned char)c)) { ++i; continue; }
        if (c == '/' && i + 1 < n && src[i + 1] == '/') {
            while (i < n && src[i] != '\n') ++i;
            continue;
        }
        if (c == '/' && i + 1 < n && src[i + 1] == '*') {
            i += 2;
            while (i + 1 < n && !(src[i] == '*' && src[i + 1] == '/')) { if (src[i] == '\n') ++line; ++i; }
            i += 2;
            continue;
        }
        if (std::isdigit((unsigned char)c)) {
            size_t j = i;
            while (j < n && (std::isalnum((unsigned char)src[j]) || src[j] == '.')) ++j;
            out.push_back({Token::Number, src.substr(i, j - i), line});
            i = j;
            continue;
        }
        if (is_ident_start(c)) {
            size_t j = i + 1;
            while (j < n && is_ident_char(src[j])) ++j;
            out.push_back({Token::Ident, src.substr(i, j - i), line});
            i = j;
            continue;
        }
        if (c == '"') {  // strings only appear in .file directives; skip them
            ++i;
            while (i < n && src[i] != '"') ++i;
            ++i;
            continue;
        }
        out.push_back({Token::Punct, std::string(1, c), line});
        ++i;
    }
    out.push_back({Token::End, "", line});
    return out;
}

bool parse_number(const std::string& s, int64_t& value, bool& is_float) {
    is_float = false;
    try {
        if (s.size() == 10 && (s[1] == 'f' || s[1] == 'F') && s[0] == '0') {
            value = (int64_t)std::stoul(s.substr(2), nullptr, 16);
            is_float = true;
            return true;
        }
        if (s.size() > 2 && s[0] == '0' && (s[1] == 'd' || s[1] == 'D')) return false;  // f64 literal
        std::string t = s;
        if (!t.empty() && (t.back() == 'U' || t.back() == 'u')) t.pop_back();
        if (t.find('.') != std::string::npos) {
            float f = std::stof(t);
            uint32_t bits;
            std::memcpy(&bits, &f, 4);
            value = bits;
            is_float = true;
            return true;
        }
        if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) {
            value = (int64_t)std::stoull(t.substr(2), nullptr, 16);
            return true;
        }
        value = (int64_t)std::stoull(t, nullptr, 10);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_type(const std::string& m, Type& t) {
    static const std::unordered_map<std::string, Type> types = {
        {"pred", Type::Pred}, {"u8", Type::U8}, {"s8", Type::S8}, {"u16", Type::U16}, {"s16", Type::S16},
        {"u32", Type::U32}, {"s32", Type::S32}, {"u64", Type::U64}, {"s64", Type::S64}, {"b8", Type::B8},
        {"b16", Type::B16}, {"b32", Type::B32}, {"b64", Type::B64}, {"f32", Type::F32}};
    auto it = types.find(m);
    if (it == types.end()) return false;
    t = it->second;
    return true;
}

bool parse_special(const std::string& s, Special& sp) {
    static const std::unordered_map<std::string, Special> m = {
        {"%tid.x", Special::TidX}, {"%tid.y", Special::TidY}, {"%tid.z", Special::TidZ},
        {"%ntid.x", Special::NtidX}, {"%ntid.y", Special::NtidY}, {"%ntid.z", Special::NtidZ},
        {"%ctaid.x", Special::CtaidX}, {"%ctaid.y", Special::CtaidY}, {"%ctaid.z", Special::CtaidZ},
        {"%nctaid.x", Special::NctaidX}, {"%nctaid.y", Special::NctaidY}, {"%nctaid.z", Special::NctaidZ},
        {"%laneid", Special::LaneId}, {"%warpid", Special::WarpId}};
    auto it = m.find(s);
    if (it == m.end()) return false;
    sp = it->second;
    return true;
}

std::vector<std::string> split_dots(const std::string& s) {
    std::vector<std::string> parts;
    std::string cur;
    for (char c : s) {
        if (c == '.') { parts.push_back(cur); cur.clear(); }
        else cur += c;
    }
    parts.push_back(cur);
    return parts;
}

class Parser {
public:
    explicit Parser(const std::string& src) : toks_(lex(src)) {}

    Module parse() {
        Module mod;
        while (peek().kind != Token::End) {
            const Token& t = peek();
            if (t.kind == Token::Ident && (t.text == ".version" || t.text == ".target" || t.text == ".address_size" ||
                                           t.text == ".file" || t.text == ".loc")) {
                skip_line();
            } else if (t.kind == Token::Ident && (t.text == ".visible" || t.text == ".weak")) {
                next();
            } else if (t.kind == Token::Ident && t.text == ".entry") {
                next();
                mod.kernels.push_back(parse_entry());
            } else if (t.kind == Token::Ident && t.text == ".extern") {
                next();
                if (peek().text == ".shared") {
                    parse_dynamic_shared_decl();
                } else {
                    skip_to(';');
                }
            } else if (t.kind == Token::Ident && t.text == ".shared") {
                throw ParseError(t.line, "module-scope .shared variables are not supported");
            } else if (t.kind == Token::Ident && (t.text == ".func" || t.text == ".global" || t.text == ".const")) {
                throw ParseError(t.line, "unsupported module-level declaration: " + t.text);
            } else {
                throw ParseError(t.line, "unexpected token at module scope: '" + t.text + "'");
            }
        }
        return mod;
    }

private:
    std::vector<Token> toks_;
    size_t pos_ = 0;
    std::vector<std::string> dynamic_shared_names_;
    // Operands naming the dynamic shared array: (instruction, operand) pairs
    // patched once the kernel's static shared size is known.
    std::vector<std::pair<size_t, int>> dynamic_refs_;

    bool is_dynamic_shared(const std::string& name) const {
        return std::find(dynamic_shared_names_.begin(), dynamic_shared_names_.end(), name) !=
               dynamic_shared_names_.end();
    }

    const Token& peek(size_t k = 0) const { return toks_[std::min(pos_ + k, toks_.size() - 1)]; }
    const Token& next() { const Token& t = toks_[pos_]; if (pos_ + 1 < toks_.size()) ++pos_; return t; }
    void expect(const std::string& s) {
        if (peek().text != s) throw ParseError(peek().line, "expected '" + s + "', got '" + peek().text + "'");
        next();
    }
    void skip_line() {
        int line = peek().line;
        while (peek().kind != Token::End && peek().line == line) next();
    }
    void skip_to(char c) {
        while (peek().kind != Token::End && peek().text != std::string(1, c)) next();
        next();
    }

    // .extern .shared .align 16 .b8 name[];
    void parse_dynamic_shared_decl() {
        expect(".shared");
        while (peek().kind == Token::Ident && peek().text[0] == '.') {
            if (peek().text == ".align") { next(); next(); } else next();
        }
        dynamic_shared_names_.push_back(next().text);
        skip_to(';');
    }

    int parse_int() {
        const Token& t = next();
        int64_t v; bool f;
        if (t.kind != Token::Number || !parse_number(t.text, v, f) || f)
            throw ParseError(t.line, "expected integer, got '" + t.text + "'");
        return (int)v;
    }

    Kernel parse_entry() {
        Kernel k;
        k.name = next().text;
        expect("(");
        int offset = 0;
        while (peek().text != ")") {
            expect(".param");
            Param p;
            Type t = Type::None;
            while (peek().kind == Token::Ident && peek().text[0] == '.') {
                std::string m = next().text;
                if (m == ".align") p.align = parse_int();
                else if (!parse_type(m.substr(1), t)) throw ParseError(peek().line, "bad param type " + m);
            }
            p.name = next().text;
            p.size = type_bytes(t);
            if (p.align == 1) p.align = p.size;
            while (peek().text == "[") { next(); p.size *= parse_int(); expect("]"); }
            offset = (offset + p.align - 1) / p.align * p.align;
            p.offset = offset;
            offset += p.size;
            k.params.push_back(p);
            if (peek().text == ",") next();
        }
        expect(")");
        k.param_bytes = offset;
        // performance directives (.maxntid, .reqntid, .minnctapersm, ...)
        while (peek().text != "{") {
            if (peek().kind == Token::End) throw ParseError(peek().line, "missing kernel body");
            next();
        }
        parse_body(k);
        return k;
    }

    struct PendingBranch { size_t instr; std::string label; int line; };

    void parse_body(Kernel& k) {
        expect("{");
        int depth = 1;
        std::unordered_map<std::string, int> labels;
        std::vector<PendingBranch> branches;
        std::unordered_map<std::string, int> params;
        for (const auto& p : k.params) params[p.name] = p.offset;

        while (depth > 0) {
            const Token& t = peek();
            if (t.kind == Token::End) throw ParseError(t.line, "unterminated kernel body");
            if (t.text == "{") { next(); ++depth; continue; }
            if (t.text == "}") { next(); --depth; continue; }
            if (t.kind == Token::Ident && peek(1).text == ":") {
                labels[t.text] = (int)k.code.size();
                next(); next();
                continue;
            }
            if (t.kind == Token::Ident && t.text[0] == '.') {
                if (t.text == ".reg") { next(); parse_reg_decl(k); continue; }
                if (t.text == ".shared") { next(); parse_shared_decl(k); continue; }
                if (t.text == ".loc" || t.text == ".file") { skip_line(); continue; }
                if (t.text == ".pragma") { skip_to(';'); continue; }
                if (t.text == ".local") throw ParseError(t.line, ".local memory is not supported");
                throw ParseError(t.line, "unsupported directive in body: " + t.text);
            }
            Instr in;
            in.line = t.line;
            if (t.text == "@") {
                next();
                if (peek().text == "!") { next(); in.guard_neg = true; }
                in.guard = reg_of(k, next());
            }
            const Token& opTok = next();
            if (opTok.kind != Token::Ident) throw ParseError(opTok.line, "expected opcode, got '" + opTok.text + "'");
            in.text = opTok.text;
            decode_opcode(opTok, in);
            std::string label;
            parse_operands(k, in, params, label);
            if (!label.empty()) branches.push_back({k.code.size(), label, in.line});
            finalize(in);
            k.code.push_back(std::move(in));
        }
        // The dynamic shared array starts after static shared memory, 16-byte aligned.
        k.uses_dynamic_shared = !dynamic_refs_.empty();
        for (const auto& ref : dynamic_refs_) k.code[ref.first].ops[ref.second].imm += dynamic_shared_base(k);
        dynamic_refs_.clear();
        for (const auto& b : branches) {
            auto it = labels.find(b.label);
            if (it == labels.end()) throw ParseError(b.line, "undefined label " + b.label);
            k.code[b.instr].target = it->second;
        }
        if (k.code.empty() || (k.code.back().op != Op::Ret && k.code.back().op != Op::Exit) || k.code.back().guard >= 0) {
            Instr r;
            r.op = Op::Ret;
            r.unit = Unit::Branch;
            r.text = "ret (implicit)";
            k.code.push_back(r);
        }
        k.reconverge_pc = compute_reconvergence(k.code);
    }

    void parse_reg_decl(Kernel& k) {
        Type t = Type::None;
        int line = peek().line;
        if (!parse_type(next().text.substr(1), t)) throw ParseError(line, "bad .reg type");
        while (true) {
            std::string name = next().text;
            if (peek().text == "<") {
                next();
                int count = parse_int();
                expect(">");
                for (int i = 0; i < count; ++i) add_reg(k, name + std::to_string(i), t);
            } else {
                add_reg(k, name, t);
            }
            if (peek().text == ",") { next(); continue; }
            break;
        }
        expect(";");
    }

    void add_reg(Kernel& k, const std::string& name, Type t) {
        if (k.reg_index.count(name)) return;
        k.reg_index[name] = (int)k.regs.size();
        k.regs.push_back({name, t});
    }

    void parse_shared_decl(Kernel& k) {
        int align = 1;
        Type t = Type::None;
        while (peek().kind == Token::Ident && peek().text[0] == '.') {
            std::string m = next().text;
            if (m == ".align") align = parse_int();
            else if (!parse_type(m.substr(1), t)) throw ParseError(peek().line, "bad .shared type " + m);
        }
        std::string name = next().text;
        int bytes = type_bytes(t);
        if (align < bytes) align = bytes;
        while (peek().text == "[") { next(); bytes *= parse_int(); expect("]"); }
        expect(";");
        int off = (k.static_shared_bytes + align - 1) / align * align;
        k.shared_symbols[name] = off;
        k.static_shared_bytes = off + bytes;
    }

    int reg_of(const Kernel& k, const Token& t) {
        auto it = k.reg_index.find(t.text);
        if (it == k.reg_index.end()) throw ParseError(t.line, "undeclared register " + t.text);
        return it->second;
    }

    void decode_opcode(const Token& tok, Instr& in) {
        auto parts = split_dots(tok.text);
        const std::string& base = parts[0];
        static const std::unordered_map<std::string, Op> ops = {
            {"ld", Op::Ld}, {"st", Op::St}, {"mov", Op::Mov}, {"cvta", Op::Cvta}, {"cvt", Op::Cvt},
            {"add", Op::Add}, {"sub", Op::Sub}, {"mul", Op::Mul}, {"mad", Op::Mad}, {"fma", Op::Fma},
            {"div", Op::Div}, {"rem", Op::Rem}, {"min", Op::Min}, {"max", Op::Max}, {"neg", Op::Neg},
            {"abs", Op::Abs}, {"and", Op::And}, {"or", Op::Or}, {"xor", Op::Xor}, {"not", Op::Not},
            {"shl", Op::Shl}, {"shr", Op::Shr}, {"setp", Op::Setp}, {"selp", Op::Selp}, {"sqrt", Op::Sqrt},
            {"rsqrt", Op::Rsqrt}, {"rcp", Op::Rcp}, {"ex2", Op::Ex2}, {"lg2", Op::Lg2}, {"bra", Op::Bra},
            {"ret", Op::Ret}, {"exit", Op::Exit}, {"bar", Op::Bar}, {"barrier", Op::Bar}, {"shfl", Op::Shfl},
            {"atom", Op::Atom}, {"red", Op::Red}, {"membar", Op::Nop}, {"fence", Op::Nop}};
        auto it = ops.find(base);
        if (it == ops.end()) throw ParseError(tok.line, "unsupported instruction: " + tok.text);
        in.op = it->second;
        int ntypes = 0;
        for (size_t i = 1; i < parts.size(); ++i) {
            const std::string& m = parts[i];
            Type t;
            if (parse_type(m, t)) {
                if (ntypes == 0) in.type = t; else in.type2 = t;
                ++ntypes;
                continue;
            }
            if (m == "f64" || m == "f16" || m == "bf16" || m == "f16x2")
                throw ParseError(tok.line, "unsupported type ." + m + " in " + tok.text);
            if (m == "global") { in.space = Space::Global; continue; }
            if (m == "shared") { in.space = Space::Shared; continue; }
            if (m == "param") { in.space = Space::Param; continue; }
            if (m == "local" || m == "const") throw ParseError(tok.line, "unsupported state space in " + tok.text);
            if (m == "to") { in.cvta_to = true; continue; }
            if (m == "v2") { in.vec = 2; continue; }
            if (m == "v4") { in.vec = 4; continue; }
            if (in.op == Op::Setp) {
                static const std::unordered_map<std::string, std::pair<Cmp, bool>> cmps = {
                    {"eq", {Cmp::Eq, false}}, {"ne", {Cmp::Ne, false}}, {"lt", {Cmp::Lt, false}},
                    {"le", {Cmp::Le, false}}, {"gt", {Cmp::Gt, false}}, {"ge", {Cmp::Ge, false}},
                    {"lo", {Cmp::Lt, true}}, {"ls", {Cmp::Le, true}}, {"hi", {Cmp::Gt, true}},
                    {"hs", {Cmp::Ge, true}}, {"equ", {Cmp::Eq, false}}, {"neu", {Cmp::Ne, false}},
                    {"ltu", {Cmp::Lt, false}}, {"leu", {Cmp::Le, false}}, {"gtu", {Cmp::Gt, false}},
                    {"geu", {Cmp::Ge, false}}};
                auto c = cmps.find(m);
                if (c != cmps.end()) { in.cmp = c->second.first; in.cmp_unsigned = c->second.second; continue; }
                if (m == "and" || m == "or" || m == "xor")
                    throw ParseError(tok.line, "setp boolean combination is not supported: " + tok.text);
            }
            if (in.op == Op::Mul || in.op == Op::Mad) {
                if (m == "lo") { in.mul = MulMode::Lo; continue; }
                if (m == "hi") { in.mul = MulMode::Hi; continue; }
                if (m == "wide") { in.mul = MulMode::Wide; continue; }
            }
            if (in.op == Op::Shfl) {
                if (m == "up") { in.shfl = ShflMode::Up; continue; }
                if (m == "down") { in.shfl = ShflMode::Down; continue; }
                if (m == "bfly") { in.shfl = ShflMode::Bfly; continue; }
                if (m == "idx") { in.shfl = ShflMode::Idx; continue; }
            }
            if (in.op == Op::Atom || in.op == Op::Red) {
                static const std::unordered_map<std::string, AtomOp> aops = {
                    {"add", AtomOp::Add}, {"min", AtomOp::Min}, {"max", AtomOp::Max}, {"exch", AtomOp::Exch},
                    {"cas", AtomOp::Cas}, {"and", AtomOp::And}, {"or", AtomOp::Or}, {"xor", AtomOp::Xor}};
                auto a = aops.find(m);
                if (a != aops.end()) { in.aop = a->second; continue; }
            }
            if (in.op == Op::Bar && m == "warp") { in.op = Op::Nop; continue; }
            // Rounding, caching and ordering qualifiers do not change functional results here.
            static const char* ignored[] = {"rn", "rz", "rm", "rp", "rni", "rzi", "rmi", "rpi", "ftz", "approx",
                                            "full", "sat", "uni", "sync", "nc", "ca", "cg", "cs", "lu", "cv",
                                            "wb", "wt", "aligned", "relaxed", "acquire", "release", "acq_rel",
                                            "gpu", "cta", "sys", "gl", "sc", "L1", "L2", "volatile", "arrive"};
            bool ok = false;
            for (const char* ig : ignored) if (m == ig) { ok = true; break; }
            if (!ok) throw ParseError(tok.line, "unsupported modifier ." + m + " in " + tok.text);
        }
        if (in.op == Op::Bar && parts.size() > 1 && parts[1] == "arrive")
            throw ParseError(tok.line, "bar.arrive is not supported");
    }

    Operand parse_operand(const Kernel& k, const std::unordered_map<std::string, int>& params, std::string& label,
                          Instr& in) {
        Operand o;
        const Token& t = peek();
        if (t.text == "[") {
            next();
            o.kind = Operand::Kind::Addr;
            const Token& b = next();
            if (b.kind == Token::Number) {
                int64_t v; bool f;
                parse_number(b.text, v, f);
                o.imm = v;
            } else if (k.reg_index.count(b.text)) {
                o.reg = k.reg_index.at(b.text);
            } else if (params.count(b.text)) {
                o.imm = params.at(b.text);
            } else if (k.shared_symbols.count(b.text)) {
                o.imm = k.shared_symbols.at(b.text);
            } else if (is_dynamic_shared(b.text)) {
                dynamic_refs_.push_back({k.code.size(), in.nops});
            } else {
                throw ParseError(b.line, "unknown address base " + b.text);
            }
            if (peek().text == "+" || peek().text == "-") {
                bool neg = next().text == "-";
                if (peek().text == "-") { next(); neg = !neg; }
                int64_t v; bool f;
                const Token& n = next();
                if (!parse_number(n.text, v, f)) throw ParseError(n.line, "bad address offset");
                o.imm += neg ? -v : v;
            }
            expect("]");
            return o;
        }
        if (t.text == "{") {
            next();
            o.kind = Operand::Kind::Vec;
            while (peek().text != "}") {
                if (o.nvec == 4) throw ParseError(t.line, "vector operand too wide");
                o.vec[o.nvec++] = reg_of(k, next());
                if (peek().text == ",") next();
            }
            next();
            return o;
        }
        bool neg = false;
        if (t.text == "-") { next(); neg = true; }
        const Token& v = next();
        if (v.kind == Token::Number) {
            int64_t x; bool f;
            if (!parse_number(v.text, x, f)) throw ParseError(v.line, "bad numeric literal " + v.text);
            o.kind = Operand::Kind::Imm;
            if (neg) {
                if (f) x ^= 0x80000000LL; else x = -x;
            }
            o.imm = x;
            return o;
        }
        Special sp;
        if (parse_special(v.text, sp)) { o.kind = Operand::Kind::Special; o.special = sp; return o; }
        auto r = k.reg_index.find(v.text);
        if (r != k.reg_index.end()) { o.kind = Operand::Kind::Reg; o.reg = r->second; return o; }
        auto s = k.shared_symbols.find(v.text);
        if (s != k.shared_symbols.end()) { o.kind = Operand::Kind::Imm; o.imm = s->second; return o; }
        if (is_dynamic_shared(v.text)) {
            o.kind = Operand::Kind::Imm;
            dynamic_refs_.push_back({k.code.size(), in.nops});
            return o;
        }
        if (in.op == Op::Bra) { label = v.text; o.kind = Operand::Kind::None; return o; }
        throw ParseError(v.line, "unknown operand " + v.text);
    }

    void parse_operands(const Kernel& k, Instr& in, const std::unordered_map<std::string, int>& params,
                        std::string& label) {
        while (peek().text != ";") {
            if (peek().kind == Token::End) throw ParseError(in.line, "missing ';'");
            if (in.nops == 5) throw ParseError(in.line, "too many operands");
            in.ops[in.nops++] = parse_operand(k, params, label, in);
            if (peek().text == "|") {  // setp/shfl second predicate destination
                next();
                in.dst2 = reg_of(k, next());
            }
            if (peek().text == ",") next();
        }
        next();
    }

    // Unit classification and scoreboard register lists.
    void finalize(Instr& in) {
        auto use = [&](const Operand& o) {
            if (o.kind == Operand::Kind::Reg || (o.kind == Operand::Kind::Addr && o.reg >= 0)) in.uses.push_back(o.reg);
            if (o.kind == Operand::Kind::Vec) for (int i = 0; i < o.nvec; ++i) in.uses.push_back(o.vec[i]);
        };
        auto def = [&](const Operand& o) {
            if (o.kind == Operand::Kind::Reg) in.defs.push_back(o.reg);
            if (o.kind == Operand::Kind::Vec) for (int i = 0; i < o.nvec; ++i) in.defs.push_back(o.vec[i]);
        };
        if (in.guard >= 0) in.uses.push_back(in.guard);
        switch (in.op) {
            case Op::St:
                for (int i = 0; i < in.nops; ++i) use(in.ops[i]);
                break;
            case Op::Red:
                for (int i = 0; i < in.nops; ++i) use(in.ops[i]);
                break;
            case Op::Bra: case Op::Ret: case Op::Exit: case Op::Bar: case Op::Nop:
                break;
            default:
                if (in.nops > 0) def(in.ops[0]);
                for (int i = 1; i < in.nops; ++i) use(in.ops[i]);
                break;
        }
        if (in.dst2 >= 0) in.defs.push_back(in.dst2);

        switch (in.op) {
            case Op::Ld: case Op::St:
                in.unit = in.space == Space::Global ? Unit::MemGlobal
                        : in.space == Space::Shared ? Unit::MemShared
                        : in.space == Space::Param  ? Unit::MemParam
                                                    : Unit::MemGeneric;
                break;
            case Op::Atom: case Op::Red: in.unit = Unit::Atomic; break;
            case Op::Div: case Op::Rem: case Op::Sqrt: case Op::Rsqrt: case Op::Rcp: case Op::Ex2: case Op::Lg2:
                in.unit = Unit::Sfu; break;
            case Op::Shfl: in.unit = Unit::MemShared; break;  // shuffles go through the shared-memory crossbar
            case Op::Bra: case Op::Ret: case Op::Exit: in.unit = Unit::Branch; break;
            case Op::Bar: in.unit = Unit::Barrier; break;
            default: in.unit = Unit::Alu; break;
        }
    }
};

}  // namespace

Module parse_ptx(const std::string& source) { return Parser(source).parse(); }

Module load_ptx_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return parse_ptx(ss.str());
}

}  // namespace ptxsim
