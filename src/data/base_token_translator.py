from abc import ABC, abstractmethod

class BaseTokenTranslator(ABC):
    '''
    Shared interface all dataset translators must implement
    '''

    @abstractmethod
    def translate(self, token_string: str) -> str:
        '''
        Translate a single token string into its natural language equivalent
        '''
        pass

    @abstractmethod
    def is_measurable_concept(self, token_string: str) -> bool:
        '''
        True if this token represents a concept that is followed by a number
        '''
        pass
        
    @abstractmethod
    def is_numeric_value(self, token_string: str) -> bool:
        '''
        True if this token represents a numeric value
        '''
        pass

    @abstractmethod
    def is_new_event_code(self, token_string: str) -> bool:
        '''
        True if this token represents a new event code
        '''
        pass    

    @abstractmethod
    def is_header_concept(self, token_string: str) -> bool:
        '''
        True if this token represents a header concept
        '''
        pass