import { Component, OnInit } from '@angular/core';
import { StoryService } from './story.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'personal-guide';
  constructor(private storyService : StoryService) { }
  ngOnInit(): void { 
    this.storyService.getStories()
  }
}
