from importlib import import_module

_LAZY_EXPORTS = {
    'LLMClient': ('src.util.llm', 'LLMClient'),
    'DeepSeekAPI': ('src.util.llm', 'DeepSeekAPI'),
    'OpenAIAPI': ('src.util.llm', 'OpenAIAPI'),
    'QualityOracle': ('src.util.eval', 'QualityOracle'),
    'GrammarCorrector': ('src.util.eval', 'GrammarCorrector'),
    'ChatFormatter': ('src.util.data', 'ChatFormatter'),
    'load_config_file': ('src.util.data', 'load_config_file'),
    'load_json_as_list': ('src.util.data', 'load_json_as_list'),
    'create_directory_for_file': ('src.util.data', 'create_directory_for_file'),
    'normalize_quotes': ('src.util.data', 'normalize_quotes'),
    'set_seed': ('src.util.env', 'set_seed'),
    'get_best_free_gpu': ('src.util.env', 'get_best_free_gpu'),
    'resolve_model_path': ('src.util.env', 'resolve_model_path'),
    'resolve_project_path': ('src.util.env', 'resolve_project_path'),
    'extract_model_name': ('src.util.env', 'extract_model_name'),
    'configure_models_root': ('src.util.env', 'configure_models_root'),
    'get_project_root': ('src.util.env', 'get_project_root'),
    'setup_logger': ('src.util.log', 'setup_logger'),
    'get_logger': ('src.util.log', 'get_logger'),
    'DatabaseManager': ('src.util.db', 'DatabaseManager'),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'LLMClient',
    'DeepSeekAPI',
    'OpenAIAPI',
    'QualityOracle',
    'GrammarCorrector',
    'ChatFormatter',
    'load_config_file',
    'load_json_as_list',
    'create_directory_for_file',
    'normalize_quotes',
    'set_seed',
    'get_best_free_gpu',
    'resolve_model_path',
    'resolve_project_path',
    'extract_model_name',
    'configure_models_root',
    'get_project_root',
    'setup_logger',
    'get_logger',
    'DatabaseManager',
]
