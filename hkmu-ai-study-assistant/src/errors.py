class StudyAssistantError(Exception): pass
class ConfigurationError(StudyAssistantError): pass
class DocumentError(StudyAssistantError): pass
class OCRUnavailableError(DocumentError): pass
class KnowledgeBaseError(StudyAssistantError): pass
class ProviderError(StudyAssistantError): pass
