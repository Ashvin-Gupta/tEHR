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

    def is_time_token(self, token_string: str) -> bool:
        t = str(token_string)
        return t.startswith("TIME ") or t.startswith("<time")

    def is_value_or_unit_token(self, token_string: str) -> bool:
        """Numeric values, quantiles, lab categories, and unit strings — not event codes."""
        t = str(token_string)
        if not t or t.startswith("<") or t == "MEDS_BIRTH":
            return False
        if self.is_time_token(t) or self.is_new_event_code(t):
            return False
        tu, tl = t.upper(), t.lower()
        if tu.startswith("Q") and len(t) <= 4 and t[1:].isdigit():
            return True
        if tl in {"low", "normal", "high", "very low", "very high"}:
            return True
        if self.is_numeric_value(t):
            return True
        if any(c in t for c in ("/", "%", "°", "µ", "μ")):
            return True
        if len(t) <= 5 and t.replace(" ", "").isalpha():
            return True
        return False