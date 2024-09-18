class StaticAnalysisError(Exception):
    """Exception raised for errors detected during static analysis."""

    def __init__(self, message: str, line: int, severity: str):
        self.message = message
        self.line = line
        self.severity = severity
        super().__init__(self.message)

    def __str__(self):
        return f"{self.severity} at line {self.line}: {self.message}"
    
class LLMKitError(Exception):
    """Base exception class for LLMKit-specific errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"LLMKitError: {self.message}"


class ImportValidationError(LLMKitError):
    """Exception raised when import validation fails."""

    def __init__(self, message: str, invalid_imports: list):
        self.invalid_imports = invalid_imports
        super().__init__(f"{message}. Invalid imports: {', '.join(invalid_imports)}")


class SyntaxFixError(LLMKitError):
    """Exception raised when syntax fixing fails."""

    def __init__(self, message: str, line: int):
        self.line = line
        super().__init__(f"{message} at line {line}")
