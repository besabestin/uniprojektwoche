export class Life {
    stories?: Story[];
}

export class Story {
    sentiment?: string;
    events?: string;
    when?: string;
    content?: string;
}

export class SearchResult {
    message?: string;
}