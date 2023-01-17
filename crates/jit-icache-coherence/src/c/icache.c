void icache__clear_cache(char *begin, char *end) {
    __builtin___clear_cache(begin, end);
    return;
}
